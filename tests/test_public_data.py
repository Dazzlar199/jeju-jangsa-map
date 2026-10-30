import pytest

from jangsa.config import DATA_PUBLIC
from jangsa.public_data import classify_license, emd_from_address, load_all, load_resident_hourly
from jangsa.schema import TIME_BANDS, region_of, to_admin

# 저장소에 들어 있는 공개 파일(이용조건 확인됨)과, 필요하면 fetch_public_data.py 로 받는 파일을 나눠 건너뛴다
HAVE_PNU = any(DATA_PUBLIC.glob("**/*시간대별_제주도민_지번*.csv"))
HAVE_BUS = any(DATA_PUBLIC.glob("**/*버스정류소현황*.csv"))
HAVE_LICENSE = any(DATA_PUBLIC.glob("**/*휴게음식점현황*.csv"))
HAVE_BC = any(DATA_PUBLIC.glob("**/*BC카드_지역별*시간대별분석_이용금액_*.csv"))


def test_address_to_emd():
    assert emd_from_address("제주특별자치도 서귀포시 성산읍 고성리 1100-13") == "성산읍"
    assert emd_from_address("제주특별자치도 서귀포시 서홍동 397-80 102호") == "서홍동"
    assert emd_from_address("제주특별자치도 서귀포시 강정동 208-5") == "대천동"          # 법정동 → 행정동
    assert emd_from_address("제주특별자치도 서귀포시 서귀동 272-18") is None             # 행정동을 정할 수 없는 법정동은 빼 둔다


def test_license_classification_and_regions():
    assert classify_license("스타벅스 서귀포점") is None and classify_license("바다커피") == "카페"
    assert classify_license("파리바게뜨 제과점") == "베이커리"
    assert region_of("노형동") == "북부" and region_of("대정읍") == "서부" and region_of("성산읍") == "동부" and region_of("중문동") == "남부"
    assert to_admin("이도이동") == "이도2동" and to_admin("제주시 노형동") == "노형동"


@pytest.mark.skipif(not HAVE_PNU, reason="도민 유동인구 파일 없음")
def test_resident_hourly_values_are_not_rounded_to_zero():
    """회귀: '유동인구'와 '유동인구 비율' 두 벌의 시각 열 중 비율 열(0.0006 꼴)이 덮어써서 소수 둘째 자리 반올림 때 읍면동 대부분이 0이 되던 결함."""
    p = load_resident_hourly()
    d = p.df
    assert set(d["time_band"]) == set(TIME_BANDS)
    assert (d["pop_avg"] > 0).all(), d[d["pop_avg"] <= 0].head()
    assert d.groupby("emd")["pop_avg"].max().gt(1).all()                  # 읍면동마다 한 시간대 이상은 1명(평균) 넘는다
    ae = d[d["emd"] == "애월읍"].set_index("time_band")["pop_avg"]
    assert len(ae) == 5 and ae.min() > 1 and ae.nunique() == 5            # 그래프가 비지 않고 시간대마다 값이 다르다
    share = ae / ae.sum() * 100
    assert abs(share.sum() - 100) < 1e-6 and share.max() < 90


@pytest.mark.skipif(not (HAVE_PNU and HAVE_BUS), reason="저장소 포함 공개 파일 없음")
def test_committed_public_loaders_work():
    got = {p.table_id: p for p in load_all()}
    assert {"resident_hourly", "bus_stops"} <= set(got)
    assert got["bus_stops"].df["stop_id"].str.len().eq(9).all()          # 공공데이터포털 9자리 ID


@pytest.mark.skipif(not (HAVE_LICENSE and HAVE_BC), reason="fetch_public_data.py 로 받는 파일 없음")
def test_fetched_public_loaders_work():
    got = {p.table_id: p for p in load_all()}
    assert {"cafe_license", "bc_tour_time", "bc_tour_dow"} <= set(got)
    t = got["bc_tour_time"].df
    assert set(t["region"]) == {"북부", "남부", "동부", "서부"} and set(t["time_band"]) == {"오전", "점심", "오후", "저녁", "밤"}
    assert (got["cafe_license"].df["store_cnt"] > 0).all()


def test_empty_public_folder_loads_nothing(tmp_path):
    assert load_all(tmp_path) == []
