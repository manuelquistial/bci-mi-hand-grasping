"use client";

import { useSettingsStore } from "@/state/settings-store";

import styles from "../settings-panel.module.css";
import {
  classifierDefaults,
  showsDepth,
  showsEstimators,
  showsNeighbors,
  showsRegularization,
} from "./classifier-options";
import { NumberField } from "./NumberField";
import { SettingsFieldset } from "./SettingsShell";

export function PreModelClassifierSection() {
  const preModel = useSettingsStore((s) => s.settings.preModel);
  const setNested = useSettingsStore((s) => s.setNested);
  const nestedEval = preModel.evaluation === "nested" || preModel.evaluation === "nested_loso";
  return (
    <SettingsFieldset legend="Classifier">
      <label className={styles.row}>
        Method
        <select
          value={preModel.classifier}
          onChange={(e) => {
            const classifier = e.target.value as typeof preModel.classifier;
            setNested("preModel", { classifier, ...classifierDefaults(classifier) });
          }}
        >
          <option value="none">None</option>
          <option value="lda">LDA</option>
          <option value="slda">Shrinkage LDA</option>
          <option value="logistic_regression">Logistic regression</option>
          <option value="linear_svm">Linear SVM</option>
          <option value="knn">k-NN</option>
          <option value="gaussian_nb">Gaussian Naive Bayes</option>
          <option value="decision_tree">Decision tree</option>
          <option value="svm_rbf">SVM-RBF</option>
          <option value="random_forest">Random forest</option>
          <option value="random_subspace_knn">Random subspace k-NN</option>
          <option value="boosted_trees">Boosted trees</option>
          <option value="bagged_trees">Bagged trees</option>
          <option value="rusboosted_trees">RUSBoosted trees</option>
        </select>
      </label>
      {showsRegularization(preModel.classifier) ? (
        <NumberField
          label="C"
          value={preModel.regularizationC ?? 1}
          step={0.1}
          min={0.0001}
          onChange={(value) => setNested("preModel", { regularizationC: value })}
        />
      ) : null}
      {preModel.classifier === "slda" ? (
        <NumberField
          label="Shrinkage"
          value={preModel.shrinkage ?? ""}
          step={0.05}
          min={0}
          max={1}
          placeholder="auto"
          onChange={(value) =>
            setNested("preModel", { shrinkage: Number.isNaN(value) ? null : value })
          }
        />
      ) : null}
      {showsNeighbors(preModel.classifier) ? (
        <NumberField
          label="Neighbors"
          value={preModel.neighbors ?? 5}
          step={1}
          min={1}
          onChange={(value) => setNested("preModel", { neighbors: value })}
        />
      ) : null}
      {preModel.classifier === "svm_rbf" ? (
        <label className={styles.row}>
          Gamma
          <input
            type="text"
            value={preModel.gamma ?? "scale"}
            onChange={(e) => setNested("preModel", { gamma: e.target.value })}
          />
        </label>
      ) : null}
      {showsDepth(preModel.classifier) ? (
        <NumberField
          label="Max depth"
          value={preModel.maxDepth ?? ""}
          step={1}
          min={1}
          placeholder="none"
          onChange={(value) =>
            setNested("preModel", { maxDepth: Number.isNaN(value) ? null : value })
          }
        />
      ) : null}
      {showsEstimators(preModel.classifier) ? (
        <NumberField
          label="Estimators"
          value={preModel.nEstimators ?? 80}
          step={1}
          min={1}
          onChange={(value) => setNested("preModel", { nEstimators: value })}
        />
      ) : null}
      {preModel.classifier === "random_subspace_knn" ? (
        <NumberField
          label="Max features"
          value={preModel.maxFeatures ?? 0.5}
          step={0.05}
          min={0.05}
          max={1}
          onChange={(value) => setNested("preModel", { maxFeatures: value })}
        />
      ) : null}
      <label className={styles.row}>
        Evaluation
        <select
          value={preModel.evaluation}
          onChange={(e) =>
            setNested("preModel", { evaluation: e.target.value as typeof preModel.evaluation })
          }
        >
          <option value="nested">Nested cross-subject</option>
          <option value="nested_loso">Nested LOSO</option>
          <option value="split">Train / validation / test</option>
          <option value="cross_validation">Cross-validation</option>
          <option value="loso">LOSO</option>
        </select>
      </label>
      {preModel.evaluation === "nested" || preModel.evaluation === "cross_validation" ? (
        <NumberField
          label="Folds"
          value={preModel.folds ?? 5}
          step={1}
          min={2}
          onChange={(value) => setNested("preModel", { folds: value })}
        />
      ) : null}
      {nestedEval ? (
        <>
          <NumberField
            label="Inner folds"
            value={preModel.innerFolds ?? 3}
            step={1}
            min={2}
            onChange={(value) => setNested("preModel", { innerFolds: value })}
          />
          <NumberField
            label="Search budget"
            value={preModel.searchBudget ?? 4}
            step={1}
            min={1}
            onChange={(value) => setNested("preModel", { searchBudget: value })}
          />
        </>
      ) : null}
      {preModel.evaluation === "split" ? (
        <>
          <NumberField
            label="Test fraction"
            value={preModel.testFraction ?? 0.2}
            step={0.05}
            min={0.05}
            max={0.8}
            onChange={(value) => setNested("preModel", { testFraction: value })}
          />
          <NumberField
            label="Validation fraction"
            value={preModel.validationFraction ?? 0.2}
            step={0.05}
            min={0.05}
            max={0.8}
            onChange={(value) => setNested("preModel", { validationFraction: value })}
          />
        </>
      ) : null}
    </SettingsFieldset>
  );
}
