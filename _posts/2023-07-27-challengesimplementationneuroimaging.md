---
layout: single
classes: wide
author_profile: true
title: "[Paper Insights] Challenges of implementing computer-aided diagnostic models for neuroimages in a clinical setting"
show_date: true
comments: true
---

[Paper link, DOI: 10.1038/s41746-023-00868-x](https://www.nature.com/articles/s41746-023-00868-x)

Despite the intense interest in using neuroimaging data (like CT or MRI scans)+machine learning/AI in the clinical setting and many studies demonstrating "promising" results, there has been limited progress on clinical implementation. The article discusses the challenges associated with the implementation of computer-aided decision (CAD) models.

<h2> Types of CAD models </h2>

FDA specifies three different CAD models:

1. Medical Image Management and Processing System (MIMPS): used to preprocess and manage radiology images
2. computer-aided triage and notification (CADt) models: used for prognosis
3. computer-aided diagnosis (CADx): used for diagnosis

<h2> Interesting challenges </h2>

1. Validation: segmentation can be verified visually. Diagnostic labels are more difficult to verify. Therefore only a handful of cases of neuroimaging CAD models are being integrated into a clinical workflow. Mostly accelerate the decision-making process due to verification issues. These CAD models are only useful for time-sensitive injuries like TBI.
2. Current CAD models have not found their place in routine clinical practice because studies are performed in research settings and concentrate on average group differences instead of single-subject classification/regression. 
3. Access to multi-modal information for better models: lack of detailed, structured, demographic, and clinical data in public imaging databases and the fact that off-the-shelf, image-to-label machine learning models are common in computer vision and thus are easier to implement than multi-input models.
4. Differential diagnosis is also understudied in single-subject classification: lack of information about the potential of CAD models to contribute meaningfully to a clinician who needs to formulate a differential diagnosis for a patient presents another translational complication. Example: data in single-subject classification studies often come with binary or categorical labels between healthy controls and different stages of Alzheimer’s. However, the clinical question is less often about whether a patient has Alzheimer’s or is cognitively normal, but whether they have a prodromal stage of Alzheimer’s or whether they have another, non-Alzheimer’s-related neurological disease (e.g. vascular or frontotemporal degeneration).
5. Beyond the current understanding of Science: seldom completely characterized by localized structural changes that would be readily identifiable by a practicing radiologist or other clinicians. Also because the association between brain structure and behavior is weaker due to limitations in the sensitivity of the imaging techniques or a pathophysiological mechanism that has no impact on brain structure, many psychiatric disorders are primarily studied using modalities that are not yet routinely acquired in clinical practice and which themselves require unique technical analysis methods.

<h2> Recommendations </h2>

1. limited rolling out of models for the identification of localized structural disorders within large and mid-sized research hospitals, trained only on local data. This is a favorable place to start with the deployment of neuroimaging CAD models because:
    1. Verification: segmentation-based models are easier to verify by humans than end-to-end diagnostic models, thus helping to catch problems with the models early on
    2. Patient harm: segmentation rather than a certain diagnosis, they do not need to operate perfectly and thus avert the risk of patient harm due to a false diagnosis but may be used to notify clinicians to possible high-risk cases in routinely collected data earlier than normal (or, such models may be used to inform a clinician in a diagnosis, e.g. by detecting a rare type of lesion)
    3. Generalizability: operating within a single institution avoids many of the problems associated with site differences.
