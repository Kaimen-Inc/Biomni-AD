## Hi, I'm Biomni-AD 🧠
#### Your AI co-scientist on the journey to conquer Alzheimer's disease.

**Biomni-AD** is developed by **Kuan-lin Huang, PhD**, building on the foundational [Biomni](https://github.com/snap-stanford/Biomni) platform by Stanford's SNAP Lab.

I can analyze omics data, mine AD knowledge databases, run bioinformatics pipelines, and help generate research hypotheses — all with full execution traces you can inspect and reproduce.

Tell me a research question to get started. The agent will always prioritize your local data lake and curated AD datasets before searching external sources.

<!-- BIOMNI_SUGGESTED_PROMPTS_START -->
**Suggested prompts based on your local data:**

*GWAS*
- *"Map the top 10 AD GWAS loci from Bellenguez 2022 (GCST90027158) to nearby genes and report their putative functions"*
- *"What are the top GWAS hits for CSF clusterin levels in the NG00052 dataset? Which of these overlap known AD risk loci?"*
- *"Extract genome-wide significant hits from the Kunkle 2019 IGAP stage-2 summary stats (NG00075) and annotate them with gene names"*

*Proteomics*
- *"Which proteins are measured across CSF, plasma, and brain tissue in the SomaScan 1.3k proteomic panel (NG00102)? Find any shared with known AD biomarkers"*

*QTL*
- *"Identify the top eQTL genes in prefrontal cortex (MFG) from NG00105 that overlap AD GWAS loci — load the cis-QTL file and filter by FDR < 0.05"*
- *"Find structural variant eQTLs in ROSMAP DLPFC (NG00118) for BIN1 and CLU — do they co-localize with GWAS signals?"*
- *"Find microglia-specific eQTLs from SingleBrain that co-localize with AD GWAS loci — load the MG top-association files"*
- *"Map isoMiGA microglia splicing QTLs (sQTLs) to the BIN1 and PTK2B loci — load union_leafcutter_top_assoc.tsv.gz"*

*Rare variants*
- *"What rare coding variants reach exome-wide significance in the ADSP European WES dataset (NG00126)?"*
- *"Run a gene-level burden analysis summary using the CHARGE/ADSP 5k WGS results (NG00165) — list top gene hits from SKAT and CMC tests"*
- *"Which coding and non-coding rare variants are most significant in African American ancestry from ADSP R3 WGS (NG00166)?"*
- *"Summarize the structural variant associations with AD risk from NG00172"*
- *"Look up all TREM2 and APOE rare variants in the RADR database (RADR_V3.xlsx) and report their clinical classifications"*

*Biomarkers*
- *"Analyze the plasma and urine biomarker data from NG00133 — which analytes differ most between AD cases and controls?"*

*Immunogenomics*
- *"Compare T-cell receptor CDR3 sequences between AD brain and blood samples using the NG00148 data"*

*Metabolomics*
- *"Identify metabolites whose MWAS weights (NG00180) are most enriched in AD-related pathways — use the EUR metabolite feature table"*

*Expression*
- *"Compare microglia gene expression (TPM) for TREM2, CX3CR1, and P2RY12 across cohorts using isoMiGA count matrices"*
<!-- BIOMNI_SUGGESTED_PROMPTS_END -->


<!-- BIOMNI_LOCAL_DATASET_SECTION_START -->
<details><summary>📊 200 data lake files available — click to expand</summary>

**Built-in Data Lake** — `/Users/kuan-lin.huang/Projects/Biomni/data/biomni_data/data_lake`

- `BindingDB_All_202409.tsv`
- `DepMap_CRISPRGeneDependency.csv`
- `DepMap_CRISPRGeneEffect.csv`
- `DepMap_Model.csv`
- `DepMap_OmicsExpressionProteinCodingGenesTPMLogp1.csv`
- `DisGeNET.parquet`
- `McPAS-TCR.parquet`
- `Virus-Host_PPI_P-HIPSTER_2020.parquet`
- `affinity_capture-ms.parquet`
- `affinity_capture-rna.parquet`
- `biomniAD/GCST90027158/GCST90027158_buildGRCh38.tsv.gz`
- `biomniAD/NG00052/README_Summary_statistics_csf_clusterin.docx`
- `biomniAD/NG00052/Summary_statistics_csf_clusterin_dataset_neurobiolaging_cruchaga_0.txt`
- `biomniAD/NG00075/NG00075_Kunkle_IGAP_SummaryStats_P-val_only_README.docx`
- `biomniAD/NG00075/NG00075_Kunkle_etal_Stage2_P-val_only_results.txt`
- `biomniAD/NG00102/Brain_SOMAscan1.3k_analyte_info.csv`
- `biomniAD/NG00102/CSF_SOMAscan1.3k_analyte_info.csv`
- `biomniAD/NG00102/Plasma_SOMAscan1.3k_analyte_info.csv`
- `biomniAD/NG00102/readme_2023_06`
- `biomniAD/NG00105/MFG_eur_expression_peer10.cis_qtl.txt.gz`
- `biomniAD/NG00105/MFG_eur_splicing_peer5_cluster.cis_qtl.txt.gz`
- `biomniAD/NG00105/MFG_eur_splicing_peer5_gene.cis_qtl.txt.gz`
- `biomniAD/NG00105/Nominal_eQTL_DD.xlsx`
- `biomniAD/NG00105/Nominal_sQTL_DD.xlsx`
- `biomniAD/NG00105/Permuted_eQTL_DD.xlsx`
- `biomniAD/NG00105/Permuted_sQTL_DD.xlsx`
- `biomniAD/NG00105/README_association_summary.pdf`
- `biomniAD/NG00105/STG_eur_expression_peer10.cis_qtl.txt.gz`
- `biomniAD/NG00105/STG_eur_splicing_peer5_cluster.cis_qtl.txt.gz`
- `biomniAD/NG00105/STG_eur_splicing_peer5_gene.cis_qtl.txt.gz`
- `biomniAD/NG00105/SVZ_eur_expression_peer5.cis_qtl.txt.gz`
- `biomniAD/NG00105/SVZ_eur_splicing_peer0_cluster.cis_qtl.txt.gz`
- `biomniAD/NG00105/SVZ_eur_splicing_peer0_gene.cis_qtl.txt.gz`
- `biomniAD/NG00105/THA_eur_expression_peer10.cis_qtl.txt.gz`
- `biomniAD/NG00105/THA_eur_splicing_peer5_cluster.cis_qtl.txt.gz`
- `biomniAD/NG00105/THA_eur_splicing_peer5_gene.cis_qtl.txt.gz`
- `biomniAD/NG00118/README_association_summary.pdf`
- `biomniAD/NG00118/SVeQTL_DD.xlsx`
- `biomniAD/NG00118/SVeQTL_MSBB_BM10_nominal.tsv.gz`
- `biomniAD/NG00118/SVeQTL_MSBB_BM22_nominal.tsv.gz`
- `biomniAD/NG00118/SVeQTL_MSBB_BM36_nominal.tsv.gz`
- `biomniAD/NG00118/SVeQTL_MSBB_BM44_nominal.tsv.gz`
- `biomniAD/NG00118/SVeQTL_Mayo_CBE_nominal.tsv.gz`
- `biomniAD/NG00118/SVeQTL_Mayo_TCX_nominal.tsv.gz`
- `biomniAD/NG00118/SVeQTL_ROSMAP_DLPFC_nominal.tsv.gz`
- `biomniAD/NG00118/SVhaQTL_DD.xlsx`
- `biomniAD/NG00118/SVhaQTL_ROSMAP_DLPFC_nominal.tsv.gz`
- `biomniAD/NG00118/SVpQTL_DD.xlsx`
- `biomniAD/NG00118/SVpQTL_ROSMAP_DLPFC_nominal.tsv.gz`
- `biomniAD/NG00118/SVsQTL_DD.xlsx`
- `biomniAD/NG00118/SVsQTL_ROSMAP_DLPFC_nominal.tsv.gz`
- `biomniAD/NG00126/ADSP_WES_EU_Belloy_2022_NIAGADS_summary_stats_p-value_only.txt`
- `biomniAD/NG00126/NG00126_README_p-value_only.txt`
- `biomniAD/NG00133/Suppl_1_plasma`
- `biomniAD/NG00133/Suppl_2_urine`
- `biomniAD/NG00133/Suppl_3_PK_Summary`
- `biomniAD/NG00148/AD_all_pchem_for_all_IR_receptor_CDR3s_Blanck_July_2022.csv`
- `biomniAD/NG00148/Huda_et_al_2022_JAD220119.pdf`
- `biomniAD/NG00148/Huda_et_al_article_Table_S2_TRA_Brain_AD_Blanck_July_2022.csv`
- `biomniAD/NG00148/Huda_et_al_article_Table_S3_TRA_Blood_AD_Blanck_July_2022.csv`
- `biomniAD/NG00165/README_CHARGE_ADSP_5k_p-valueonly.txt`
- `biomniAD/NG00165/adsp.charge.r1.wgs.all.pooled.CMC_0.01_EPACTS.2024_0221.snv_indels_p-valueonly.txt`
- `biomniAD/NG00165/adsp.charge.r1.wgs.all.pooled.CMC_0.05_EPACTS.2024_0221.snv_indels_p-valueonly.txt`
- `biomniAD/NG00165/adsp.charge.r1.wgs.all.pooled.SKAT_0.01_EPACTS.2024_0221.snv_indels_p-valueonly.txt`
- `biomniAD/NG00165/adsp.charge.r1.wgs.all.pooled.SKAT_0.05_EPACTS.2024_0221.snv_indels_p-valueonly.txt`
- `biomniAD/NG00165/adsp.charge.r1.wgs.all.pooled.STAAR.2024_0221.snv_indels_p-valueonly.txt`
- `biomniAD/NG00166/ADSP.r3.wgs.aa.codingSTAARresults.2024.0303.txt`
- `biomniAD/NG00166/ADSP.r3.wgs.aa.noncoding.noncodingSTAARresults.2024.0303.txt`
- `biomniAD/NG00166/ADSP.r3.wgs.his.codingSTAARresults.2024.0303.txt`
- `biomniAD/NG00166/ADSP.r3.wgs.his.noncoding.noncodingSTAARresults.2024.0303.txt`
- `biomniAD/NG00166/ADSP.r3.wgs.nhw.codingSTAARresults.2024.0303.txt`
- `biomniAD/NG00166/ADSP.r3.wgs.nhw.noncoding.noncodingSTAARresults.2024.0303.txt`
- `biomniAD/NG00166/ADSP.r3.wgs.pp.codingSTAARresults.2024.0303.txt`
- `biomniAD/NG00166/ADSP.r3.wgs.pp.noncodingSTAARresults.2024.0303.txt`
- `biomniAD/NG00166/README_ADSP_R3_WGS_P-valueOnly.txt`
- `biomniAD/NG00172/SVs_p-valueOnly.txt`
- `biomniAD/NG00172/readme_p-valOnly.txt`
- `biomniAD/NG00180/MWASweights_AFRmetab.zip`
- `biomniAD/NG00180/MWASweights_EURmetab.zip`
- `biomniAD/NG00180/NG00180_readme.txt`
- `biomniAD/NG00180/ST03_EURprotFeatureInfo.csv`
- `biomniAD/NG00180/ST04_AFRprotFeatureInfo.csv`
- `biomniAD/NG00180/ST05_EURmetabFeatureInfo.csv`
- `biomniAD/NG00180/ST06_AFRmetabFeatureInfo.csv`
- `biomniAD/NG00180/metabFeatureInfo_data_dictionary.xlsx`
- `biomniAD/NG00180/protFeatureInfo_data_dictionary.xlsx`
- `biomniAD/RADR/RADR_V3.xlsx`
- `biomniAD/SingleBrain/Ast1_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/Ast2_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/Ast3_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/Ast4_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/Ast_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/End_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/Ext1_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/Ext2_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/Ext3_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/Ext4_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/Ext5_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/Ext6_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/Ext7_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/Ext8_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/Ext_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/IN1_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/IN2_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/IN3_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/IN4_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/IN5_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/IN6_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/IN7_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/IN_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/MG1_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/MG2_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/MG3_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/MG4_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/MG_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/MiGA3_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/OD1_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/OD2_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/OD3_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/OD_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/OPC1_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/OPC2_eqtl_top_assoc.tsv.gz`
- `biomniAD/SingleBrain/OPC_eqtl_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/GENCODE_SUPPA_A3_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/GENCODE_SUPPA_A5_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/GENCODE_SUPPA_AF_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/GENCODE_SUPPA_AL_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/GENCODE_SUPPA_RI_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/GENCODE_SUPPA_SE_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/GENCODE_expression_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/GENCODE_leafcutter_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/GENCODE_transcript_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/union_SUPPA_A3_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/union_SUPPA_A5_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/union_SUPPA_AF_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/union_SUPPA_AL_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/union_SUPPA_RI_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/union_SUPPA_SE_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/union_expression_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/union_leafcutter_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_QTL/union_transcript_top_assoc.tsv.gz`
- `biomniAD/isoMiGA_counts/gaffney_gene_counts.gencode.tsv.gz`
- `biomniAD/isoMiGA_counts/gaffney_gene_counts.union.tsv.gz`
- `biomniAD/isoMiGA_counts/gaffney_gene_tpm.gencode.tsv.gz`
- `biomniAD/isoMiGA_counts/gaffney_gene_tpm.union.tsv.gz`
- `biomniAD/isoMiGA_counts/gaffney_transcript_counts.gencode.tsv.gz`
- `biomniAD/isoMiGA_counts/gaffney_transcript_counts.union.tsv.gz`
- `biomniAD/isoMiGA_counts/gaffney_transcript_tpm.gencode.tsv.gz`
- `biomniAD/isoMiGA_counts/gaffney_transcript_tpm.union.tsv.gz`
- `biomniAD/isoMiGA_counts/ipsc_gene_counts.gencode.tsv.gz`
- `biomniAD/isoMiGA_counts/ipsc_gene_counts.union.tsv.gz`
- `biomniAD/isoMiGA_counts/ipsc_gene_tpm.gencode.tsv.gz`
- `biomniAD/isoMiGA_counts/ipsc_gene_tpm.union.tsv.gz`
- `biomniAD/isoMiGA_counts/ipsc_transcript_counts.gencode.tsv.gz`
- `biomniAD/isoMiGA_counts/ipsc_transcript_counts.union.tsv.gz`
- `biomniAD/isoMiGA_counts/ipsc_transcript_tpm.gencode.tsv.gz`
- `biomniAD/isoMiGA_counts/ipsc_transcript_tpm.union.tsv.gz`
- `biomniAD/isoMiGA_counts/raj_gene_counts.gencode.tsv.gz`
- `biomniAD/isoMiGA_counts/raj_gene_counts.union.tsv.gz`
- `biomniAD/isoMiGA_counts/raj_gene_tpm.gencode.tsv.gz`
- `biomniAD/isoMiGA_counts/raj_gene_tpm.union.tsv.gz`
- `biomniAD/isoMiGA_counts/raj_transcript_counts.gencode.tsv.gz`
- `biomniAD/isoMiGA_counts/raj_transcript_counts.union.tsv.gz`
- `biomniAD/isoMiGA_counts/roussos_gene_counts.gencode.tsv.gz`
- `biomniAD/isoMiGA_counts/roussos_gene_counts.union.tsv.gz`
- `biomniAD/isoMiGA_counts/roussos_gene_tpm.gencode.tsv.gz`
- `biomniAD/isoMiGA_counts/roussos_gene_tpm.union.tsv.gz`
- `biomniAD/isoMiGA_counts/roussos_transcript_counts.gencode.tsv.gz`
- `biomniAD/isoMiGA_counts/roussos_transcript_counts.union.tsv.gz`
- `biomniAD/isoMiGA_counts/roussos_transcript_tpm.gencode.tsv.gz`
- `biomniAD/isoMiGA_counts/roussos_transcript_tpm.union.tsv.gz`
- `broad_repurposing_hub_molecule_with_smiles.parquet`
- `broad_repurposing_hub_phase_moa_target_info.parquet`
- `co-fractionation.parquet`
- `czi_census_datasets_v4.parquet`
- `ddinter_alimentary_tract_metabolism.csv`
- `ddinter_antineoplastic.csv`
- `ddinter_antiparasitic.csv`
- `ddinter_blood_organs.csv`
- `ddinter_dermatological.csv`
- `ddinter_hormonal.csv`
- `ddinter_respiratory.csv`
- `ddinter_various.csv`
- `dosage_growth_defect.parquet`
- `enamine_cloud_library_smiles.pkl`
- `evebio_assay_table.csv`
- `evebio_bundle_table.csv`
- `evebio_compound_table.csv`
- `evebio_control_table.csv`
- `evebio_detailed_result_table.csv`
- `evebio_observed_points_table.csv`
- `evebio_summary_result_table.csv`
- `evebio_target_table.csv`
- `gene_info.parquet`
- `genebass_missense_LC_filtered.pkl`
- `genebass_pLoF_filtered.pkl`
- `genebass_synonymous_filtered.pkl`
- `genetic_interaction.parquet`
- `go-plus.json`
- `gtex_tissue_gene_tpm.parquet`

</details>
<!-- BIOMNI_LOCAL_DATASET_SECTION_END -->
