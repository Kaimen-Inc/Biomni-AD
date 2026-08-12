# AD Workbench Data Catalog

This document provides a summary of the datasets available from the AD Workbench (Alzheimer's Disease Data Initiative). When encountering these datasets locally, agents should refer to these descriptions to understand their contents, modalities, and appropriate use cases.

## Clinical and Observational Cohorts
*   **University of Kansas Alzheimer's Disease Research Center - GNPC (545 Patients):** Longitudinal demographic, clinical, cognitive, and proteomic data on participants ranging from no cognitive impairment to dementia.
*   **TOMMORROW Study:** Clinical trial dataset (metadata available on GAAIN) with user guide and infographic.
*   **The GERAS Studies (US, Japan, EU, II) and Harmonized Cohort:** Prospective observational studies across multiple regions assessing societal costs, resource use, and caregiver burden associated with AD dementia. The combined/harmonized dataset includes variables from these cohorts.
*   **The Caerphilly Prospective Study (2959 Patients):** Epidemiological study on lifestyle factors, cardiovascular disease, and eventually stroke, cognitive function, and disability in men.
*   **Fox Insight (54614 Patients):** Online, longitudinal health study of people with and without Parkinson's disease, including patient-reported outcomes and genetic data.
*   **Bio-Hermes (1000 Patients):** Platform study comparing blood and digital biomarker tests with brain amyloid PET scans and traditional cognitive tests.
*   **Alzheimer's Disease and Healthy Aging Data (250937 Patients):** CDC surveillance data on health and well-being indicators for older adults.

## Transcriptomics, Epigenomics & Multi-omics (RNA-seq, ATAC-seq, ChIP-seq, Methylation)
*   **APOE isoforms human microglia xenotransplantation model:** ATAC-seq and RNA-seq profiling the epigenomic and transcriptomic landscapes of human microglia with different APOE isoforms (APOE2, APOE3, APOE4).
*   **RNAseq in Alzheimer's Disease patients - GSE53697 (17 Patients):** RNAseq analysis tracking RNA level changes during AD progression from the Mount Sinai Brain Bank.
*   **NPH Integrative Analysis Datasets:** Single-nucleus atlas from cortical biopsies of living individuals with early AD pathology, defining the Early Cortical Amyloid Response.
*   **Mis-spliced transcripts in TDP-43-related ALS/FTD (15 Patients):** Coordinated transcriptomic and proteomic studies of TDP-43 depleted iPSC-derived neurons.
*   **Genome-wide H3K27ac profiles (GSE102538):** Acetylomic variation in post-mortem entorhinal cortex tissue.
*   **Genome-wide DNA methylation profiling (GSE76105, 68 Patients):** Methylation screen of the superior temporal gyrus (STG) revealing epigenetic signatures of AD.
*   **Brain Cell Type-Specific Enhancer-Promoter Interactome Maps:** ATAC-seq and PLAC-seq studying noncoding regulatory regions and microglia enhancers in sporadic AD.
*   **BORCS6 KD transcriptomics on iNeurons & CRISPRi Screens:** RNA-seq and CRISPRi screens on iPSC-derived neurons.
*   **CUT&Tag benchmarking:** Benchmarking of CUT&Tag against ChIP-seq profiles for histone modifications.

## Biomarkers and Diagnostics (Blood, Plasma, Serum)
*   **Plasma microRNA biomarker detection for MCI - GSE90828 (23 Patients):** Differential correlation analysis of microRNAs to identify MCI markers.
*   **Blood based 12-miRNA signature - GSE46579 (70 Patients):** NGS of miRNAs from blood samples differentiating AD from unaffected controls and other CNS illnesses.
*   **Detection of AD at MCI using Autoantibodies - GSE74763 (100 Patients):** Human protein microarrays used to identify differentially expressed autoantibody biomarkers in serum.
*   **BCG-PANDA (49 Patients):** Study on the BCG vaccine effects on plasma amyloid peptide ratio and Amyloid Probability Score.

## Synthetic Datasets
*   **Synthetic EPAD dataset (200 Patients):** Modeled after the European Prevention of Alzheimer's Dementia project; contains biomarker, cognition, socio-demographic, and imaging data.
*   **DPUK Synthetic Dataset (150618 Patients):** Synthetic dataset spanning survey, biomarker, and imaging variables, modeled from three population cohorts.
*   **Heart Failure Synthetic (1000 Patients):** Subset from Get With The Guidelines Heart Failure.

## Specialized Data (EEG, Qualitative, Standards)
*   **EEG recordings from AD, FTD and Healthy subjects (88 Patients):** Resting-state closed-eyes EEG recordings.
*   **Digital Medicine Society Mixed Method Survey (1007 Patients):** Qualitative and quantitative survey on meaningful aspects of health in ADRD.
*   **ADDI Standard Variables:** Mapping of 124 commonly used ADRD variables to C-Surv and CDISC standards.

## Usage Guide for the Agent
*   **Is it structured or unstructured?** This catalog contains clinical tabular data (e.g., GERAS, synthetic cohorts), raw and processed sequencing data (RNA/ATAC/ChIP-seq), biomarker matrices, and time-series data (EEG).
*   **How to query?** Depending on the dataset:
    *   For clinical & synthetic sets, use pandas to explore patient-level distributions and harmonize columns (check the ADDI Standard Variables).
    *   For omics data, expect raw count matrices or processed bigWig/bed/bam files. Standard bioinformatics pipelines (scanpy, DESeq2) apply.
    *   For EEG data, use MNE-Python for signal processing.
