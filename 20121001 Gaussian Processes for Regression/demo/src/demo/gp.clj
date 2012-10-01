(ns demo.gp
  (:use (incanter core stats))
  (:require [incanter.charts :as charts]))
(set! *warn-on-reflection* true)


(defn RBF
  "Calculates the covariance matrix using simplified squared exponential function."
  [X1 X2]
  (let [Σ (for [i (range (count X1))
                j (range (count X2))]
            (let [d (minus (nth X1 i) (nth X2 j))
                  d (if (seq? d) (sum d) d)] ;;accepts matrices and vectors

              (exp (mult -1/2 (pow d 2)))))]

    ;; build matrix representation
    (matrix (partition (count X2) Σ))))


(defn GP
  "Predictive mean and variance using noise-free observations given by the posterior."
  [k X Y X*]
  (let [K (k X X)
        K-1 (solve K)
        f* (mmult (k X* X) K-1 Y)]
    {:m f*
     :var (minus (k X* X*) (mmult (trans (k X X*)) K-1 (k X X*)))}))






(defn mvrnorm
  "NOTE: incanter/sample-mvn uses cholesky, which requires positive semi def. So we use eigenvalue decomposition instead."
  [size & {:keys [mean sigma]}]
  (let [p (count mean)
        eS (decomp-eigenvalue sigma)
        X (matrix (sample-normal (* size p)) p)
        values (map (fn [x] (if (< x 0) 0 (sqrt x))) (:values eS))]

    (plus mean (mmult (:vectors eS)
                      (diag values)
                      (trans X)))))


(defn plot-samples
  "Sample form normal distributing with infered mean and covariance."
  [n {f* :m cov* :var} X Y X*]
  (let [plot (charts/xy-plot X* f*)]
    (dotimes [i n] (charts/add-lines plot X* (mvrnorm 1 :mean f* :sigma cov*)))
    (charts/add-points plot X Y)
    (view plot)))


;; examples from the gp book
(def X (matrix [-4 -3 -1 0 2]))
(def Y (matrix [-2 0 1 2 -1]))
(def X* (matrix  (range -5 5 0.2040816)))

;; (view (charts/scatter-plot X Y))
;; (plot-samples 5 (GP RBF X Y X*) X Y X*)

;; TODO:
;; exemplo usando dados de venda do Walmart
;; exemplo usando dados da Petro