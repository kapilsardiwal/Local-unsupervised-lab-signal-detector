"""NHANES 2017-March 2020 pre-pandemic labs -> ADLB-style long format.

Source: CDC/NCHS, public domain. https://wwwn.cdc.gov/nchs/nhanes/
Outputs data/processed/nhanes_long.csv.gz (USUBJID, PARAMCD, AVAL) and
nhanes_subjects.csv (demographics + self-reported conditions, used only for validation).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/nhanes"
OUT = ROOT / "data/processed"

LABS = {
    "P_BIOPRO": {"LBXSATSI": "ALT", "LBXSAL": "ALB", "LBXSAPSI": "ALP", "LBXSASSI": "AST", "LBXSC3SI": "BICARB",
                 "LBXSBU": "BUN", "LBXSCLSI": "CL", "LBXSCK": "CK", "LBXSCR": "CREAT", "LBXSGB": "GLOBUL",
                 "LBXSGL": "GLUC", "LBXSGTSI": "GGT", "LBXSIR": "IRON", "LBXSKSI": "K", "LBXSLDSI": "LDH",
                 "LBXSOSSI": "OSMOL", "LBXSPH": "PHOS", "LBXSNASI": "SODIUM", "LBXSTB": "BILI", "LBXSCA": "CA",
                 "LBXSCH": "CHOL", "LBXSTP": "PROT", "LBXSTR": "TRIG", "LBXSUA": "URATE"},
    "P_CBC": {"LBXWBCSI": "WBC", "LBDLYMNO": "LYM", "LBDMONO": "MONO", "LBDNENO": "NEUT", "LBDEONO": "EOS",
              "LBDBANO": "BASO", "LBXRBCSI": "RBC", "LBXHGB": "HGB", "LBXHCT": "HCT", "LBXMCVSI": "MCV",
              "LBXMC": "MCHC", "LBXMCHSI": "MCH", "LBXRDW": "RDW", "LBXPLTSI": "PLAT", "LBXMPSI": "MPV"},
    "P_GHB": {"LBXGH": "HBA1C"},
    "P_HDL": {"LBDHDD": "HDL"},
    "P_HSCRP": {"LBXHSCRP": "CRP"},
    "P_ALB_CR": {"URDACT": "ACR"},
}


def read(name):
    return pd.read_sas(RAW / f"{name}.xpt", format="xport").set_index("SEQN")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    demo = read("P_DEMO")
    adults = demo.index[demo["RIDAGEYR"] >= 18]

    wide = pd.concat([read(f)[list(m)].rename(columns=m) for f, m in LABS.items()], axis=1)
    wide = wide.loc[wide.index.intersection(adults)]
    wide = wide[wide[list(LABS["P_BIOPRO"].values())].notna().mean(axis=1) > 0.8]  # had the chemistry panel

    long = wide.rename_axis("USUBJID").reset_index().melt(id_vars="USUBJID", var_name="PARAMCD", value_name="AVAL")
    long = long.dropna(subset=["AVAL"])
    long["USUBJID"] = long["USUBJID"].astype(int).astype(str)
    long.to_csv(OUT / "nhanes_long.csv.gz", index=False)

    diq, kiq, mcq = read("P_DIQ"), read("P_KIQ_U"), read("P_MCQ")
    yes = lambda df, c: (df[c] == 1).reindex(wide.index, fill_value=False) if c in df else False
    subj = pd.DataFrame({
        "age": demo["RIDAGEYR"], "sex": demo["RIAGENDR"].map({1: "M", 2: "F"}),
        "pregnant": demo["RIDEXPRG"] == 1,
    }).reindex(wide.index)
    subj["diabetes"] = yes(diq, "DIQ010")
    subj["insulin"] = yes(diq, "DIQ050")
    subj["kidney_failure"] = yes(kiq, "KIQ022")
    subj["dialysis"] = yes(kiq, "KIQ025")
    subj["liver_condition"] = yes(mcq, "MCQ160L") | yes(mcq, "MCQ500")
    # multi-select items store the option code (3 = cirrhosis, 4 = viral hepatitis)
    subj["cirrhosis"] = (mcq["MCQ510C"] == 3).reindex(wide.index, fill_value=False)
    subj["viral_hepatitis"] = (mcq["MCQ510D"] == 4).reindex(wide.index, fill_value=False)
    subj["heart_failure"] = yes(mcq, "MCQ160B")
    subj["cancer"] = yes(mcq, "MCQ220")
    subj.index = subj.index.astype(int).astype(str)
    subj.rename_axis("USUBJID").to_csv(OUT / "nhanes_subjects.csv")
    print(f"adults with labs: {len(wide):,}; records: {len(long):,}; analytes: {long.PARAMCD.nunique()}")
    print(subj.drop(columns=["age", "sex"]).sum().to_string())


if __name__ == "__main__":
    main()
