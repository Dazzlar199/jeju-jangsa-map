import hashlib

import pandas as pd

from jangsa.schema import EMD_CITY, INDUSTRIES, TABLES
from jangsa.synthetic import LABEL_COL, SEED, generate, to_center_format, write_all


def _digest(frames):
    h = hashlib.sha256()
    for k in sorted(frames):
        h.update(k.encode())
        h.update(pd.util.hash_pandas_object(frames[k], index=False).values.tobytes())
    return h.hexdigest()


def test_same_seed_same_data_and_other_seed_differs():
    assert _digest(generate(SEED)) == _digest(generate(SEED))
    assert _digest(generate(SEED)) != _digest(generate(SEED + 1))


def test_files_are_byte_identical_across_runs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_all(a)
    write_all(b)
    for p in sorted(a.glob("*.csv")):
        assert hashlib.md5(p.read_bytes()).hexdigest() == hashlib.md5((b / p.name).read_bytes()).hexdigest(), p.name


def test_43_emd_and_industries_present():
    f = generate()
    assert len(EMD_CITY) == 43 and sum(1 for c in EMD_CITY.values() if c == "제주시") == 26
    assert set(f["card_emd"]["emd"]) == set(EMD_CITY) and set(f["card_emd"]["industry"]) == set(INDUSTRIES)


def test_every_file_is_marked_synthetic_and_center_format(syn_dir):
    for tid, sp in TABLES.items():
        if not sp.synthetic_file:
            continue
        p = syn_dir / sp.synthetic_file
        assert p.name.startswith("가상_")
        df = pd.read_csv(p, encoding="utf-8-sig", nrows=5)
        assert LABEL_COL in df.columns and (df[LABEL_COL] == "가상").all()
    card = pd.read_csv(syn_dir / TABLES["card_emd"].synthetic_file, encoding="utf-8-sig", nrows=3, dtype=str)
    assert {"base_year_month", "bjd", "biz_cd", "biz_nm", "user_type", "use_cnt", "use_amt"} <= set(card.columns)
    assert card["base_year_month"].iloc[0] == "202410"


def test_plausible_distribution():
    f = generate()
    c = f["card_emd"]
    c = c[c.industry == "카페"].assign(mo=lambda d: d.ym.str[5:])
    tour = c[c.emd == "성산읍"].groupby("mo").sales_amt.sum()
    assert tour["08"] > tour["02"] * 1.3                       # 관광지: 여름이 겨울보다 높다
    d = f["card_dow"]
    d = d[(d.emd == "성산읍") & (d.industry == "카페")].groupby("dow").sales_amt.sum()
    assert d["토"] > d["월"]                                   # 관광지: 주말이 높다
    small = f["store_emd"]
    assert (small[small.emd.isin(["추자면", "우도면"])].store_cnt < 10).any()    # 소수 셀이 실제로 생긴다


def test_roundtrip_through_csv_profile_recovers_canonical(store):
    f = generate()
    got = store.con.execute("SELECT COUNT(*), SUM(sales_amt) FROM card_emd").fetchone()
    assert got[0] == len(f["card_emd"])
    assert abs(got[1] - f["card_emd"]["sales_amt"].sum()) / got[1] < 0.001      # 법정동·이용자구분으로 쪼갰다 합쳐도 같다
