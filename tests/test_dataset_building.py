import pandas as pd

from src.dataset_building import filter_dataset_by_country, merge_datasets_by_school

def test_filter_dataset_by_country_keeps_only_requested_country():

    df = pd.DataFrame({"CNT": ["Switzerland","France","Switzerland"], "student_id": [1, 2, 3]})

    result = filter_dataset_by_country(df, "Switzerland")

    assert len(result) == 2
    assert result["CNT"].eq("Switzerland").all()
    assert set(result["student_id"].tolist()) == {1, 3}


def test_filter_dataset_by_country():
    df_student = pd.DataFrame({"CNT": ["Switzerland", "France", "Switzerland", "Germany","Estonia"], "student_id": [1, 2, 3, 4, 5]})
    filtered_df = filter_dataset_by_country(df_student, "Switzerland")
    assert len(filtered_df) == 2
    assert set(filtered_df["student_id"].tolist()) == {1, 3}


def test_merge_datasets_by_school():
    student_df = pd.DataFrame({"CNTSCHID": [3100174, 36000293, 38000278], "CNTSTUID": [3107947, 3619705, 3800485]})
    school_df = pd.DataFrame({"CNTSCHID": [36000293, 3100174, 38000278], "SCHLTYPE": ["Private independent", "Private Government-dependent", "Public"]})
    merged_df = merge_datasets_by_school(student_df, school_df)
    assert len(merged_df) == 3
    assert set(merged_df["CNTSTUID"].tolist()) == {3107947, 3619705, 3800485}
    assert set(merged_df["SCHLTYPE"].tolist()) == {"Private Government-dependent", "Private independent", "Public"}
    assert set(merged_df["CNTSCHID"].tolist()) == {36000293, 3100174, 38000278}


def test_filter_dataset_by_country_does_not_modify_original_dataframe():
    df = pd.DataFrame({"CNT": ["Switzerland", "France"], "PV1MATH": [500, 450]})
    original = df.copy(deep=True)
    filter_dataset_by_country(df,"Switzerland",)

    pd.testing.assert_frame_equal(df, original)


def test_merge_datasets_by_school_preserves_students_rows():

    students = pd.DataFrame({"CNTSCHID": [10, 20, 30], "CNTSTUID": [101, 102, 103]})
    schools = pd.DataFrame({"CNTSCHID": [10, 20],"SCHLTYPE": ["Public", "Private"]})

    result = merge_datasets_by_school(students, schools)

    assert len(result) == len(students)
    assert set(result["CNTSTUID"].tolist()) == {101, 102, 103}

    missing_school_type = result.loc[result["CNTSCHID"] == 30, "SCHLTYPE"].iloc[0]
    assert pd.isna(missing_school_type)


def test_merge_datasets_by_school_adds_school_features():

    students = pd.DataFrame({"CNTSCHID": [10, 20, 22],"CNTSTUID": [101, 102, 103]})
    schools = pd.DataFrame({"CNTSCHID": [10, 20], "SCHLTYPE": ["Public", "Private"]})

    result = merge_datasets_by_school(students,schools)

    assert result.loc[result["CNTSTUID"] == 101, "SCHLTYPE"].iloc[0] == "Public"
    assert result.loc[result["CNTSTUID"] == 102, "SCHLTYPE"].iloc[0] == "Private"
    assert pd.isna(result.loc[result["CNTSTUID"] == 103, "SCHLTYPE"].iloc[0])

