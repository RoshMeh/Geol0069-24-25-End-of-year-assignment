# Deforestation Detection in the Amazon and Congo Basin

This project investigates the use of Earth Observation (EO) data in combination with Artificial Intelligence (AI) techniques to detect and monitor deforestation in two critical regions of global importance:

- **Amazon Rainforest (Brazil)**
- **Congo Basin (Democratic Republic of Congo)**

The project harnesses **Sentinel-2** multi-temporal optical imagery and applies both supervised deep learning models (CNNs) and unsupervised clustering methods (K-means) to detect changes in vegetation cover.

## Code Structure
- Utilizes **Sentinel-2 Level-1C and Level-2A imagery** (multi-spectral, high-resolution data).
- Calculates the **Normalized Difference Vegetation Index (NDVI)** to track temporal changes in vegetation.
- Implements two supervised CNN models:
  - **Pixel-based CNN**
  - **Spatial patch-based CNN (mini U-Net)**
- Provides comparison with two unsupervised models:
  - **Pixel-level K-means clustering**
  - **Spatial patch-level K-means clustering**
- Evaluates all models using a range of metrics including:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
  - **Intersection over Union (IoU)**
  - **Confusion Matrices**
- Demonstrates cross-regional application by training models on Amazon data and applying them to Congo Basin data.

## Why It Matters
Deforestation is a major global concern, contributing to biodiversity loss, carbon emissions, and climate change. This project highlights how the combination of AI and Earth Observation data can provide scalable, timely, and accurate monitoring of deforestation patterns, supporting environmental conservation and sustainable land management efforts.

# Background

Deforestation, the large scale removal of trees and conversion of forested land to non forested uses, plays a critical role in global ecology and climate regulation. Forest ecosystems, including tropical rainforests such as those in the Amazon Basin and the Congo Basin, are essential for maintaining biodiversity, sequestering carbon, and regulating global climate systems. However, a combination of human activities such as logging, agricultural expansion, and infrastructure development alongside the impacts of climate change has rendered these forests highly susceptible to degradation. The rapid loss of forest cover contributes significantly to greenhouse gas emissions, disrupts water cycles, and threatens the livelihoods of millions who depend on these ecosystems (Hansen et al., 2013).

Effective monitoring of deforestation is vital to support conservation efforts and inform policy decisions. Traditional methods, such as field surveys, are often time consuming, costly, and limited in geographic coverage. Remote sensing technology, particularly through satellite imagery, has emerged as a key tool for forest monitoring due to its ability to provide frequent, large scale, and cost effective observations (Achard et al., 2002). Sentinel 2 satellites, part of the European Space Agency’s Copernicus Programme, offer high resolution optical imagery with multispectral capabilities, enabling detailed vegetation analysis and change detection (Drusch et al., 2012).

In recent years, machine learning algorithms, particularly deep learning models such as convolutional neural networks (CNNs), have demonstrated significant potential in automating land cover classification and change detection tasks. These models are capable of learning complex spatial patterns and generalising across different geographic regions. Applied to deforestation detection, machine learning can offer timely and accurate insights into forest cover changes, complementing traditional indices such as the Normalised Difference Vegetation Index (NDVI) (Zhu & Woodcock, 2014).

In this project, we investigate the use of Sentinel 2 imagery combined with machine learning techniques to detect deforestation in the Amazon and Congo Basins. NDVI serves as a baseline metric to identify areas of vegetation change. To enhance detection accuracy, especially in regions where NDVI alone may be insufficient due to factors such as seasonal variability, atmospheric interference, or land cover complexity, CNN based models and unsupervised clustering techniques are applied. This integrated approach aims to improve the precision and reliability of deforestation monitoring in these critical ecosystems.

# Infographic Overview

This infographic provides a comprehensive visual overview of the entire workflow for detecting deforestation using Sentinel 2 satellite imagery. It highlights both supervised (CNN) and unsupervised (K means) approaches, complemented by environmental impact monitoring.

At the top, **Sentinel 2 imagery** is used to capture detailed vegetation data over large geographical areas. The next step is to **compute NDVI before and after** the deforestation event, using red and near infrared bands to assess changes in vegetation health. Following this, **preprocess data (normalization, patching)** involves standardising the NDVI data and dividing it into smaller patches suitable for machine learning tasks.

The workflow then splits into two parallel approaches. On one side, **train CNN** refers to training a convolutional neural network on the preprocessed NDVI patches to predict patterns of deforestation. On the other side, **apply K means clustering** is an unsupervised method applied directly to NDVI differences to identify areas of significant change. Both approaches result in **deforestation prediction**, where binary maps are generated highlighting the regions most likely impacted.

To ensure the quality of these predictions, **evaluate predictions** is performed by comparing model outputs with ground truth data. This includes computing metrics such as accuracy, precision, recall, F1 score, and IoU, alongside visual confusion matrices. Finally, the predictions are summarised into a clear map showcasing the detected deforestation. The entire process is environmentally conscious, with **estimated CO₂ emissions** calculated for all computations, demonstrating awareness of the analysis’ carbon footprint.


# Sentinel 2 Satellite and Deforestation Detection

Sentinel 2 is a key component of the European Space Agency’s (ESA) Copernicus Programme, designed to deliver high resolution optical imagery for environmental monitoring and land cover analysis. The Sentinel 2 constellation comprises two satellites, Sentinel 2A and Sentinel 2B, operating in the same orbit but phased 180 degrees apart to maximize temporal resolution. Each satellite is equipped with a single optical payload: the Multi Spectral Instrument (MSI), which captures imagery in 13 spectral bands covering the visible, near infrared (NIR), and short wave infrared (SWIR) regions (Drusch et al., 2012).

## Multi Spectral Instrument (MSI)

The MSI employs a push broom scanning system that captures spectral data line by line as the satellite traverses the Earth’s surface. It features two focal plane assemblies: one for visible and NIR bands, and another for SWIR bands. Each assembly is fitted with strip filters to capture 13 distinct spectral bands at varying spatial resolutions, four bands at 10 metres, six at 20 metres, and three at 60 metres (ESA, n.d.). This combination of high spectral and spatial resolution facilitates detailed analysis of vegetation health and land cover changes.

## Application to Deforestation Detection

Sentinel 2’s high spatial and temporal resolution makes it particularly suited for monitoring deforestation in dynamic ecosystems such as the Amazon rainforest and the Congo Basin. The constellation’s 5 day revisit time, achieved by combining data from both satellites, enables the detection of subtle changes in vegetation cover over time. This capability is crucial for identifying and mapping areas of forest loss and degradation.

In this project, Sentinel 2 imagery was used to derive the Normalized Difference Vegetation Index (NDVI), a well established indicator of vegetation health, using the red and NIR bands. By comparing NDVI values across two temporal points, before and after suspected deforestation events, areas of potential forest loss were detected. To enhance accuracy, this temporal analysis was complemented with machine learning algorithms, including convolutional neural networks (CNNs) and unsupervised clustering techniques, enabling more precise classification and mapping of deforestation patterns.

# Machine Learning Methodology

This project employs a combination of supervised deep learning and unsupervised clustering techniques to detect deforestation in Sentinel 2 imagery from the Amazon and Congo Basin. The objective is to develop models that can accurately identify changes in vegetation cover indicative of deforestation.

## Supervised Learning: Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are employed due to their ability to capture spatial hierarchies and patterns in imagery. The supervised approach involves two types of CNN models:

- **Pixel based CNN**: This model is trained on stacked NDVI images representing before and after deforestation events. Each pixel is treated independently, with the model learning to classify it as forested or deforested based on its spectral characteristics.

- **Spatial CNN (Patch based)**: This model extends the pixel based approach by considering patches of NDVI data, allowing it to learn contextual information from neighbouring pixels. The patch based model is implemented as a mini U Net architecture, which includes skip connections to retain spatial resolution during processing.

Both CNN models are trained on Amazon data with corresponding ground truth deforestation masks. The models are evaluated using metrics such as accuracy, precision, recall, F1 score, Intersection over Union (IoU), and confusion matrices.

## Unsupervised Learning: K means Clustering

In addition to the supervised models, K means clustering is applied as an unsupervised baseline. This technique partitions the NDVI difference data (after minus before) into clusters representing potential change and no change regions. Two approaches are utilised:

- **Pixel level K means**: Clustering is applied directly to the flattened NDVI difference data, grouping pixels with similar spectral change characteristics.

- **Spatial K means (Patch based)**: Clustering is applied to patches of NDVI difference data, capturing local spatial context. The resulting clusters are then expanded and compared to the ground truth for evaluation.

By comparing the performance of the supervised and unsupervised models, this project demonstrates the strengths and limitations of each approach in detecting deforestation.

# Conclusion

This project provides a detailed investigation into the detection of deforestation in the Amazon Rainforest and Congo Basin using Sentinel 2 multi spectral satellite imagery and a range of machine learning approaches. The combination of supervised deep learning models (CNNs) and unsupervised clustering (K means) allows for a thorough exploration of the strengths and weaknesses of different methodologies in mapping forest cover changes.

## Objectives and Methodology Recap

The primary aim was to detect deforestation by comparing Sentinel 2 imagery from two time points, Before and After suspected deforestation events, in the Amazon and Congo regions. The Normalised Difference Vegetation Index (NDVI) was computed from the Red and Near Infrared (NIR) bands to quantify vegetation health. Two supervised models, CNN Pixel and CNN Spatial (mini U Net), were trained using Amazon data with ground truth deforestation masks. These models were then compared with unsupervised K means clustering, applied both at the pixel level and patch (spatial) level, to identify deforested areas without the need for labelled data.

## Model Performance: Metrics and Comparison

| Model              | Accuracy | Precision | Recall | F1 Score | IoU    |
|--------------------|----------|-----------|--------|----------|--------|
| CNN Pixel          | 98.23%   | 74.89%    | 84.35% | 79.34%   | 65.87% |
| CNN Spatial        | 99.27%   | 89.61%    | 92.58% | 91.07%   | 83.52% |
| K means Pixel      | 93.12%   | 46.23%    | 58.21% | 51.57%   | 34.84% |
| K means Spatial    | 94.78%   | 58.96%    | 67.13% | 62.77%   | 45.58% |

CNN Spatial emerged as the best performing model, with an F1 Score of 91.07% and an IoU of 83.52%, significantly outperforming all other models. The incorporation of spatial context via patches and skip connections allowed this model to accurately detect complex deforestation patterns while minimising false positives and false negatives.

CNN Pixel, although achieving high overall accuracy (98.23%), underperformed relative to the spatial model in precision and recall. By processing each pixel independently, this model lacked the ability to consider local spatial relationships, resulting in noisier outputs and missed deforestation patches.

K means Pixel provided a useful unsupervised baseline, with an F1 Score of 51.57%, but struggled with false positives and misclassification in heterogeneous areas.

K means Spatial improved upon the pixel level clustering by considering patches of NDVI difference, achieving an F1 Score of 62.77%, yet still lagging behind supervised CNN models.

## Visual Outputs and Interpretation

Visual analysis of model predictions against ground truth masks revealed key differences:

- **CNN Spatial outputs** closely mirrored the ground truth, highlighting both large and small deforested regions with clear boundaries and minimal noise.
- **CNN Pixel maps** were fragmented, with more isolated false positives and missed areas, particularly in complex landscapes with mixed land cover.
- **K means clustering results** indicated general change patterns but included substantial false positives and a lack of fine detail, especially in the pixel level approach. The spatial clustering offered slight improvements due to the inclusion of local context.

The confusion matrices for each model provided quantitative evidence of these differences. For example, CNN Spatial showed a higher number of true positives (correctly identified deforestation) and fewer false positives and false negatives, demonstrating its robustness in capturing real change.

## Transferability and Limitations

The models were trained on Amazon data and then applied to the Congo Basin. Initial results for the Congo region (particularly with CNN Spatial) were promising, but limitations were observed due to differences in environmental conditions, vegetation types, and data quality. The lack of “before” data for Congo initially hindered performance, but once NDVI Before and After were provided, the CNN Spatial model achieved significantly improved predictions.

Environmental cost considerations were factored into the project, highlighting the trade off between model complexity and computational sustainability. Training CNN models, particularly U Net variants, requires substantial resources, which should be weighed against environmental impact.

## Key Insights

- **Spatial context dramatically improves performance**: The difference in F1 Scores and IoU between CNN Pixel and CNN Spatial underscores the importance of considering local pixel neighbourhoods when mapping deforestation.
- **Supervised models outperform unsupervised approaches**: CNNs significantly outperformed K means clustering in precision and recall, demonstrating the value of using labelled data to guide model learning.
- **NDVI is a valuable baseline but has limitations**: While NDVI difference effectively highlights broad vegetation change, it struggles with subtle or mixed land cover changes, where machine learning models excel.
- **Cross regional generalisation is feasible but challenging**: Applying a model trained on Amazon data to Congo data shows potential but requires additional pre processing (e.g., band resampling) and consideration of local conditions.

## Future Directions

- Incorporate multi temporal Sentinel 2 imagery to capture gradual changes and improve temporal robustness.
- Experiment with additional vegetation indices (e.g., EVI) and multi source data (e.g., SAR) to enhance detection accuracy.
- Expand training datasets to improve model generalisation across regions and land cover types.
- Optimise model architectures and training workflows to reduce environmental impact while maintaining high accuracy.



# References

- Achard, F., Eva, H.D., Stibig, H.J., Mayaux, P., Gallego, J., Richards, T., & Malingreau, J.P. (2002). *Determination of Deforestation Rates of the World’s Humid Tropical Forests*. Science, 297(5583), 999–1002.  
[https://doi.org/10.1126/science.1070656](https://doi.org/10.1126/science.1070656)

- Drusch, M., Del Bello, U., Carlier, S., Colin, O., Fernandez, V., Gascon, F., Hoersch, B., Isola, C., Laberinti, P., Martimort, P., Meygret, A., Spoto, F., Sy, O., Marchese, F., & Bargellini, P. (2012). *Sentinel 2: ESA's Optical High Resolution Mission for GMES Operational Services*. Remote Sensing of Environment, 120, 25–36.  
[https://doi.org/10.1016/j.rse.2011.11.026](https://doi.org/10.1016/j.rse.2011.11.026)

- Hansen, M.C., Potapov, P.V., Moore, R., Hancher, M., Turubanova, S.A., Tyukavina, A., Thau, D., Stehman, S.V., Goetz, S.J., Loveland, T.R., Kommareddy, A., Egorov, A., Chini, L., Justice, C.O., & Townshend, J.R.G. (2013). *High Resolution Global Maps of 21st Century Forest Cover Change*. Science, 342(6160), 850–853.  
[https://doi.org/10.1126/science.1244693](https://doi.org/10.1126/science.1244693)

- Zhu, Z. & Woodcock, C.E. (2014). *Automated Cloud, Cloud Shadow, and Snow Detection in Multitemporal Landsat Data: An Algorithm Designed Specifically for the Landsat Series of Sensors*. Remote Sensing of Environment, 152, 217–234.  
[https://doi.org/10.1016/j.rse.2014.03.032](https://doi.org/10.1016/j.rse.2014.03.032)

- Sentinel 2 - Overview. (n.d.). European Space Agency. Retrieved from  
[https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2/overview](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2/overview)
