behprepost_map = {
    0: "No",
    1: "Yes, pre-surgery only",
    2: "Yes, post-surgery only",
    3: "Yes, pre- and post-surgery",
    4: "Yes, no surgery"
}

diag_basis_map = {
    1: "Clinical examination only (history and physical exam)",
    2: "Clinical-diagnostic tests or exploratory surgery/autopsy (no microscopic confirmation)",
    4: "Specific biochemical and/or immunological laboratory tests",
    5: "Hematological or cytological confirmation (primary tumor or metastases); unclear if cytology or histology",
    6: "Histological confirmation of metastases only (including autopsy)",
    7: "Histological confirmation of the primary tumor (or unknown whether primary or metastases); possibly autopsy with histological confirmation"
}

diffgrad_map = {
    1: "Well differentiated (low-grade)",
    2: "Moderately differentiated (intermediate)",
    3: "Poorly differentiated (high-grade)",
    4: "Undifferentiated / anaplastic",
    9: "Unknown / not applicable / not determined"
}

behavior_map = {
    2: "In situ",
    3: "Malignant"
}

gender_map = {
    1: "Male",
    2: "Female"
}

her2_stat_map = {
    0: "0 (negative)",
    1: "1+ (negative)",
    2: "2+ (equivocal)",
    3: "3+ (positive)",
    4: "Not determined",
    7: "Not determined (7)",
    9: "Not assessable / unknown"
}

hr_stat_map = {
    0: "Negative",
    1: "Positive",
    9: "Not assessable / unknown"
}

mari_uitslag_map = {
    1: "MARI node negative",
    2: "ITC (≤ 0.2 mm)",
    3: "Micrometastasis (>0.2 - ≤2 mm)",
    4: "MARI node positive",
    5: "MARI node not removed",
    9: "Unknown outcome"
}

tumor_type_map = {
    501300: "Invasive breast carcinoma",
    502200: "Ductal carcinoma in situ",
    503200: "Lobular carcinoma in situ"
    # Add additional codes if you have them
}

neeja_map = {
    0: "No",
    1: "Yes"
}

neejaonb_map = {
    0: "No",
    1: "Yes",
    9: "Unknown"
}

swk_map = {
    0: "No",
    1: "Yes",
    8: "Not registered in this region"
}

vital_status_map = {
    0: "Alive",
    1: "Deceased"
}

later_map = {
    "1.0": "Left",
    "2.0": "Right",
    "X": "Unknown"
}

swk_uitslag_map = {
    1: "Negative sentinel node",
    2: "ITC (≤ 0.2 mm)",
    3: "Micrometastasis (>0.2 - ≤2 mm)",
    4: "Positive sentinel node (>2 mm)",
    9: "Sentinel node not found"
}

topo_sublok_map = {
    "C500": "Nipple/areola",
    "C501": "Central portion of breast",
    "C502": "Upper-inner quadrant",
    "C503": "Lower-inner quadrant",
    "C504": "Upper-outer quadrant",
    "C505": "Lower-outer quadrant",
    "C506": "Axillary extension (tail)",
    "C508": "Overlapping area of breast",
    "C509": "Breast, NOS"
}

morphology_map = {
    8000: "Neoplasm, NOS",
    8001: "Malignant tumor cells",
    8004: "Malignant tumor, spindle cell type",
    8010: "Carcinoma, NOS",
    8012: "Large cell carcinoma, NOS",
    8013: "Large cell neuroendocrine carcinoma",
    8020: "Undifferentiated carcinoma, NOS",
    8022: "Pleomorphic carcinoma",
    8030: "Giant cell and spindle cell carcinoma",
    8032: "Spindle cell carcinoma, NOS",
    8033: "Pseudosarcomatous carcinoma",
    8035: "Carcinoma with osteoclast-like giant cells",
    8041: "Small cell carcinoma, NOS",
    8045: "Mixed small and large cell carcinoma",
    8046: "Non-small cell carcinoma",
    8070: "Squamous cell carcinoma, NOS",
    8071: "Keratinizing squamous cell carcinoma",
    8074: "Spindle cell squamous carcinoma",
    8082: "Lymphoepithelial carcinoma",
    8140: "Adenocarcinoma, NOS",
    8141: "Scirrhous adenocarcinoma",
    8145: "Diffuse type adenocarcinoma",
    8200: "Adenoid cystic carcinoma",
    8201: "Cribriform carcinoma",
    8211: "Tubular adenocarcinoma",
    8230: "Solid carcinoma, NOS",
    8240: "Neuroendocrine tumor, NOS / Grade 1 (carcinoid)",
    8244: "Mixed adenoneuroendocrine carcinoma (MANEC)",
    8246: "Neuroendocrine carcinoma, NOS",
    8249: "Neuroendocrine tumor, grade 2/3 (atypical carcinoid)",
    8255: "Adenocarcinoma with mixed subtypes",
    8260: "Papillary adenocarcinoma, NOS",
    8290: "Oncocytic adenoma / carcinoma (Hurthle cell carcinoma)",
    8310: "Clear cell adenocarcinoma, NOS",
    8314: "Lipid-rich carcinoma",
    8315: "Glycogen-rich carcinoma",
    8401: "Apocrine adenocarcinoma",
    8407: "Microcystic adnexal carcinoma / Sclerosing sweat gland carcinoma",
    8410: "Sebaceous gland adenocarcinoma",
    8430: "Mucoepidermoid carcinoma",
    8441: "Serous cystadenocarcinoma, NOS",
    8470: "Mucinous cystadenocarcinoma, NOS",
    8480: "Mucinous adenocarcinoma",
    8481: "Mucin-producing adenocarcinoma",
    8490: "Signet ring cell carcinoma / 'poorly cohesive' carcinoma",
    8500: "Ductal carcinoma, NOS",
    8501: "Comedocarcinoma, NOS",
    8502: "Secretory carcinoma",
    8503: "Intraductal papillary adenocarcinoma",
    8504: "Encapsulated (intracystic) papillary carcinoma",
    8507: "Intraductal micropapillary carcinoma",
    8508: "Cystic hypersecretory carcinoma",
    8509: "Solid papillary carcinoma",
    8510: "Medullary carcinoma, NOS",
    8512: "Medullary carcinoma with lymphoid stroma",
    8513: "Atypical medullary carcinoma",
    8514: "Ductal carcinoma, desmoplastic type",
    8519: "Pleomorphic lobular carcinoma in situ",
    8520: "Lobular carcinoma, NOS",
    8521: "Ductular carcinoma",
    8522: "Ductal and lobular carcinoma",
    8523: "Ductal carcinoma mixed with another carcinoma type",
    8524: "Lobular carcinoma mixed with another carcinoma type",
    8530: "Inflammatory carcinoma",
    8540: "Paget's disease of the breast",
    8541: "Paget's disease + invasive ductal carcinoma",
    8543: "Paget's disease + intraductal carcinoma (DCIS)",
    8550: "Acinar cell carcinoma",
    8560: "Adenosquamous carcinoma",
    8562: "Epithelial-myoepithelial carcinoma",
    8570: "Adenocarcinoma with squamous metaplasia",
    8571: "Adenocarcinoma with cartilaginous or bony metaplasia",
    8572: "Adenocarcinoma with spindle cell metaplasia",
    8573: "Adenocarcinoma with apocrine metaplasia",
    8574: "Adenocarcinoma with neuroendocrine differentiation",
    8575: "Metaplastic carcinoma, NOS",
    8980: "Carcinosarcoma, NOS",
    8982: "Myoepithelial carcinoma",
    8983: "Malignant adenomyoepithelioma"
}

therapie_map = {
    100000: "Surgery, NOS",
    120000: "Local tumor resection",
    "130C50": "Breast-conserving surgery, NOS",
    "131C50": "Lumpectomy (without axillary lymph node dissection)",
    "132C50": "Lumpectomy (with axillary lymph node dissection)",
    "140C50": "Non-breast-conserving surgery, NOS",
    "141C50": "Mastectomy (without axillary lymph node dissection)",
    "142C50": "Mastectomy (with axillary lymph node dissection)",
    190000: "Resection for another indication (incidental finding)",
    315000: "Regional lymph node dissection for metastases",
    690100: "Surgical treatment abroad"
}

FEATURE_PROMPT_LABELS = {
    "leeft": "Age",
    "gesl": "Gender",
    "incjr": "IncidentYear",
    "tumsoort": "TumorType",
    "diag_basis": "DiagnosisBasis",
    "topo_sublok": "TopographySub",
    "later": "Laterality",
    "morf": "Morphology",
    "gedrag": "Behavior",
    "diffgrad": "DifferentiationGrade",
    "stadium": "Stage",
    "ond_lymf": "NumLymphExamined",
    "pos_lymf": "NumLymphPositive",
    "er_stat": "ERstatus",
    "pr_stat": "PRstatus",
    "her2_stat": "HER2status",
    "dcis_comp": "DCIScomponent",
    "multifoc": "Multifocality",
    "tum_afm": "TumorSize(mm)",
    "swk": "SentinelProcedure",
    "swk_uitslag": "SentinelOutcome",
    "okd": "AxillaryDissection",
    "org_chir": "OrganSurgery",
    "uitgebr_chir_code": "SurgeryCode",
    "dir_reconstr": "DirectReconstruction",
    "chemo": "Chemo",
    "target": "TargetTherapy",
    "horm": "HormoneTherapy",
    "rt": "Radiotherapy",
    "meta_rt": "RTMetastases",
    "meta_chir": "SurgeryMetastases"
}
FEATURES_TO_INCLUDE = list(FEATURE_PROMPT_LABELS.keys())