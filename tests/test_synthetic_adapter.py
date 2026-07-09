from adapters.synthetic import DEFAULT_DATA_PATHS, SyntheticAdapter


def test_default_paths_point_at_repo_root_excel_workbooks():
    assert DEFAULT_DATA_PATHS["AWS"].name == "aws_finops_data.xlsx"
    assert DEFAULT_DATA_PATHS["Azure"].name == "azure_finops_data.xlsx"
    assert DEFAULT_DATA_PATHS["AWS"].exists()
    assert DEFAULT_DATA_PATHS["Azure"].exists()


def test_loads_only_rows_for_its_own_provider():
    aws = SyntheticAdapter("AWS")
    azure = SyntheticAdapter("Azure")
    assert set(aws.df["provider"].unique()) == {"AWS"}
    assert set(azure.df["provider"].unique()) == {"Azure"}


def test_drops_trailing_blank_excel_rows():
    aws = SyntheticAdapter("AWS")
    assert aws.df["resource_id"].isna().sum() == 0
    assert aws.df["date"].isna().sum() == 0


def test_status_is_the_new_uniform_running_stopped_scheme():
    aws = SyntheticAdapter("AWS")
    azure = SyntheticAdapter("Azure")
    assert set(aws.df["status"].unique()) <= {"running", "stopped"}
    assert set(azure.df["status"].unique()) <= {"running", "stopped"}
    # EC2/VM are the only archetype with real stopped instances in this dataset.
    assert "stopped" in set(aws.df.loc[aws.df["service"] == "EC2", "status"])
    assert "stopped" in set(azure.df.loc[azure.df["service"] == "Virtual Machine", "status"])
