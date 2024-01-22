use ndarray::prelude::*;
use numpy::{PyArray1, PyArray2};
use powerboxesrs::iou::parallel_iou_distance;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::borrow::Cow;

#[pyclass(subclass)]
struct HuaRs {
    iou_threshold: f32,
    score_threshold: f32,
}

impl HuaRs {
    fn filter_bounding_boxes<'a>(
        &self,
        bounding_boxes: &'a Array2<f32>,
        class_probabilities: &'a Array2<f32>,
        uncertainty_scores: &'a Array1<f32>,
        scales: &'a Array1<i64>,
    ) -> (
        Cow<'a, Array2<f32>>,
        Cow<'a, Array2<f32>>,
        Cow<'a, Array1<f32>>,
        Cow<'a, Array1<i64>>,
    ) {
        // Filter out the bounding boxes with low scores if score_threshold is provided.
        if self.score_threshold != 0.0 {
            let max_prob = class_probabilities.map_axis(Axis(1), |view| {
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

            let filtered_boxes = Cow::Owned(bounding_boxes.select(Axis(0), mask.as_slice()));
            let filtered_probabilities = Cow::Owned(class_probabilities.select(Axis(0), mask.as_slice()));
            let filtered_uncertainties = Cow::Owned(uncertainty_scores.select(Axis(0), mask.as_slice()));
            let filtered_scales = Cow::Owned(scales.select(Axis(0), mask.as_slice()));
    
            (
                filtered_boxes,
                filtered_probabilities,
                filtered_uncertainties,
                filtered_scales,
            )
        }
        // If score_threshold is not provided, then return the original arrays
        else {
            (
                Cow::Borrowed(bounding_boxes),
                Cow::Borrowed(class_probabilities),
                Cow::Borrowed(uncertainty_scores),
                Cow::Borrowed(scales),
            )
        }
    }

    fn group_bounding_boxes(&self, filtered_boxes: &Array2<f32>) -> Vec<Vec<usize>> {
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

    fn class_level_aggregation(&self, scores: &Vec<f32>) -> f32 {
        scores.iter().sum()
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

    fn accumulate_scores(
        &self,
        group: &Vec<usize>,
        uncertainties: &Array1<f32>,
        scales: &Array1<i64>,
        scores_tree_dict: &HashMap<i64, Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        let mut scores_tree = scores_tree_dict.clone();

        for &idx in group {
            scores_tree
                .entry(scales[idx])
                .or_insert_with(Vec::new)
                .push(uncertainties[idx]);
        }

        scores_tree.into_values().collect()
    }

    fn build_scores_tree(
        &self,
        groups: &Vec<Vec<usize>>,
        uncertainties: &Array1<f32>,
        scales: &Array1<i64>,
    ) -> Vec<Vec<Vec<f32>>> {
        assert_eq!(scales.len(), uncertainties.len());

        let mut scores_tree_dict = HashMap::new();

        for scale in scales.iter().cloned() {
            scores_tree_dict.entry(scale).or_insert_with(Vec::new);
        }

        let scores_tree: Vec<Vec<Vec<f32>>> = groups
            .into_iter()
            .map(|group| self.accumulate_scores(&group, &uncertainties, &scales, &scores_tree_dict))
            .collect();

        scores_tree
    }

    fn aggregate_uncertainties_tree(
        &self,
        groups: &Vec<Vec<usize>>,
        scores_list: &Vec<Vec<Vec<f32>>>,
    ) -> f32 {
        assert_eq!(groups.len(), scores_list.len());

        let informativeness_score = self.object_level_aggregation(
            &scores_list
                .iter()
                .map(|object_scores| {
                    object_scores
                        .iter()
                        .map(|class_scores| self.class_level_aggregation(&class_scores))
                        .collect::<Vec<f32>>()
                })
                .collect::<Vec<Vec<f32>>>()
                .iter()
                .map(|scale_scores| self.scale_level_aggregation(&scale_scores))
                .collect::<Vec<f32>>(),
        );

        informativeness_score
    }

    fn run(
        &self,
        bounding_boxes: &Array2<f32>,
        class_probabilities: &Array2<f32>,
        uncertainty_scores: &Array1<f32>,
        scales: &Array1<i64>,
    ) -> (f32, Option<Vec<Vec<usize>>>) {
        let (filtered_bounding_boxes, _, filtered_uncertainties, filtered_scales) = self
            .filter_bounding_boxes(
                &bounding_boxes,
                &class_probabilities,
                &uncertainty_scores,
                &scales,
            );

        let groups = self.group_bounding_boxes(&filtered_bounding_boxes);
        let scores_tree =
            self.build_scores_tree(&groups, &filtered_uncertainties, &filtered_scales);

        let informativeness_score = self.aggregate_uncertainties_tree(&groups, &scores_tree);

        (informativeness_score, None)
    }
}

#[pyclass(extends=HuaRs, subclass)]
struct HUA {}

#[pymethods]
impl HUA {
    #[new]
    fn new(iou_threshold: f32, score_threshold: f32) -> (Self, HuaRs) {
        (
            HUA {},
            HuaRs {
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
            &bounding_boxes,
            &class_probabilities,
            &uncertainty_scores,
            &scales,
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
