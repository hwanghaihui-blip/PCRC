from pcrc.data.credit import download_credit_dataset, load_credit_frame


def test_credit_loader():
    archive = download_credit_dataset()
    frame = load_credit_frame(archive)
    assert "default_next_month" in frame.columns
    assert "ID" not in frame.columns
    assert len(frame) == 30000
