use ndarray::prelude::*;
use numpy::{PyArray1, PyArray2};
use powerboxesrs::iou::parallel_iou_distance;
use pyo3::prelude::*;
use std::{collections::HashMap, io};

pub mod test;

#[pyclass(subclass)]
struct HUA_RS {
    iou_threshold: f32,
    score_threshold: f32,
}

impl HUA_RS {
    fn new(iou_threshold: f32, score_threshold: f32) -> Self {
        Self {
            iou_threshold,
            score_threshold,
        }
    }

    fn filter_bounding_boxes(
        &self,
        bounding_boxes: Array2<f32>,
        class_probabilities: Array2<f32>,
        uncertainty_scores: Array1<f32>,
        scales: Array1<i64>,
    ) -> (Array2<f32>, Array2<f32>, Array1<f32>, Array1<i64>) {
        // Filter out the bounding boxes with low scores if score_threshold is provided.
        if self.score_threshold != 0.0 {
            let max_prob = class_probabilities.map_axis(Axis(0), |view| {
                *view
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
            });
            let mask: Vec<usize> = max_prob
                .iter()
                .enumerate()
                .filter(|(_, &x)| x >= self.score_threshold)
                .map(|(i, _)| i)
                .collect();

            let filtered_boxes = bounding_boxes.select(Axis(0), mask.as_slice());
            let filtered_probabilities = class_probabilities.select(Axis(0), mask.as_slice());
            let filtered_uncertainties = uncertainty_scores.select(Axis(0), mask.as_slice());
            let filtered_scales = scales.select(Axis(0), mask.as_slice());

            return (
                filtered_boxes,
                filtered_probabilities,
                filtered_uncertainties,
                filtered_scales,
            );
        }
        // If score_threshold is not provided, then return the original tensors
        else {
            return (
                bounding_boxes,
                class_probabilities,
                uncertainty_scores,
                scales,
            );
        }
    }

    fn group_bounding_boxes(&self, filtered_boxes: Array2<f32>) -> Vec<Vec<usize>> {
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let ious: Array2<f32> =
            1.0 - parallel_iou_distance(&filtered_boxes, &filtered_boxes).mapv(|x| x as f32);

        for i in 0..filtered_boxes.shape()[0] {
            let mut group_found = false;
            let iou_with_groups = ious.row(i).mapv(|x| x >= self.iou_threshold);
            for (group_idx, group) in groups.iter().enumerate() {
                if group.iter().any(|&i| iou_with_groups[i]) {
                    groups[group_idx].push(i);
                    group_found = true;
                    break;
                }
            }
            if !group_found {
                groups.push(vec![i]);
            }
        }

        // println!("groups: {:?}", groups);
        groups
    }

    fn class_level_aggregation(&self, scores: Vec<Vec<f32>>) -> Vec<f32> {
        scores.iter().map(|row| row.iter().sum()).collect()
    }

    fn scale_level_aggregation(&self, scores: &Vec<f32>) -> f32 {
        *scores
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn object_level_aggregation(&self, scores: &Vec<f32>) -> f32 {
        scores.iter().sum()
    }

    fn aggregate_group_scores(
        &self,
        group: &Vec<usize>,
        scale_scores_dict: &mut HashMap<i64, Vec<f32>>,
        uncertainties: &Array1<f32>,
        scales: &Array1<i64>,
    ) -> Vec<Vec<f32>> {
        // Each group, i.e. all the bboxes assigned to a particular object,
        // has to aggregate its own scale-level scores
        let mut scale_scores = scale_scores_dict.clone();

        // Group all the uncertainty scores for each bbox in the current
        // `group` at the scale level
        for &idx in group {
            // Get scale of idx and append the uncertainty score to the
            // corresponding scale
            scale_scores.entry(scales[idx]).or_default().push(uncertainties[idx]);
        }

        scale_scores.values().cloned().collect()
    }

    fn aggregate_uncertainties_at_scale_level(
        &self,
        groups: &Vec<Vec<usize>>,
        uncertainties: &Array1<f32>,
        scales: &Array1<i64>,
    ) -> Vec<Vec<f32>> {

        assert_eq!(scales.len(), uncertainties.len());
    
        let mut scale_level_scores: Vec<Vec<f32>> = Vec::new();
        let mut scale_scores_dict: HashMap<i64, Vec<f32>> = scales.iter().map(|scale| (*scale, vec![])).collect();
    
        let scores_list: Vec<Vec<Vec<f32>>> = groups.iter().map(|group| {
                                self.aggregate_group_scores(&group, &mut scale_scores_dict, &uncertainties, &scales)
                            }).collect();

    
        for scores in scores_list {
            scale_level_scores.push(self.class_level_aggregation(scores));
        }

        // println!("Scale level scores: {:?}", scale_level_scores);
        scale_level_scores
    }

    fn aggregate_uncertainties_at_object_level(
        &self,
        groups: &Vec<Vec<usize>>,
        scale_level_scores: &Vec<Vec<f32>>,
    ) -> Vec<f32> {
        assert_eq!(groups.len(), scale_level_scores.len());

        let obj_lvl_scores = scale_level_scores
        .iter()
        .map(|scores| self.scale_level_aggregation(scores))
        .collect();
        // println!("obj_lvl_scores: {:?}", obj_lvl_scores);
        obj_lvl_scores
        
    }

    fn aggregate_uncertainties_at_image_level(
        &self,
        groups: &Vec<Vec<usize>>,
        object_level_scores: &Vec<f32>,
    ) -> f32 {
        assert_eq!(groups.len(), object_level_scores.len());

        let informativeness = self.object_level_aggregation(object_level_scores);
        informativeness
    }

    fn run(
        &self,
        bounding_boxes: Array2<f32>,
        class_probabilities: Array2<f32>,
        uncertainty_scores: Array1<f32>,
        scales: Array1<i64>,
    ) -> (f32, Vec<Vec<usize>>) {
        let (filtered_bounding_boxes, _, filtered_uncertainties, filtered_scales) = self
            .filter_bounding_boxes(
                bounding_boxes.clone(),
                class_probabilities.clone(),
                uncertainty_scores.clone(),
                scales.clone(),
            );

        let groups = self.group_bounding_boxes(filtered_bounding_boxes.clone());

        let scale_level_scores = self.aggregate_uncertainties_at_scale_level(
            &groups,
            &filtered_uncertainties,
            &filtered_scales,
        );

        let object_level_scores =
            self.aggregate_uncertainties_at_object_level(&groups, &scale_level_scores);

        let informativeness_score =
            self.aggregate_uncertainties_at_image_level(&groups, &object_level_scores);

        (informativeness_score, groups)
    }
}

#[pyclass(extends=HUA_RS, subclass)]
struct HUA {}

#[pymethods]
impl HUA {
    #[new]
    fn new(iou_threshold: f32, score_threshold: f32) -> (Self, HUA_RS) {
        (
            HUA {},
            HUA_RS {
                iou_threshold,
                score_threshold,
            },
        )
    }

    fn run(
        self_: PyRef<'_, Self>,
        bounding_boxes: &PyArray2<f32>,
        class_probabilities: &PyArray2<f32>,
        uncertainty_scores: &PyArray1<f32>,
        scales: &PyArray1<i64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let super_ = self_.as_ref(); // Get &BaseClass
        let py = self_.py();
        let bounding_boxes = unsafe { bounding_boxes.as_array() }.to_owned();
        let class_probabilities = unsafe { class_probabilities.as_array() }.to_owned();
        let uncertainty_scores = unsafe { uncertainty_scores.as_array() }.to_owned();
        let scales = unsafe { scales.as_array() }.to_owned();
        let result = super_.run(
            bounding_boxes,
            class_probabilities,
            uncertainty_scores,
            scales,
        );
        let informativeness_score = result.0.into_py(py);
        let groups = result.1.into_py(py);
        return Ok((informativeness_score, groups));
    }
}

#[pymodule]
fn hua_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<HUA>()?;
    Ok(())
}
