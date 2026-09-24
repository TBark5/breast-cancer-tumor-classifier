from streamlit.testing.v1 import AppTest
from src.utils import ROOT


def test_dashboard_navigation_and_sample_interaction():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert len(app.metric) == 4
    for page in ["Model performance", "Feature analysis", "Explainable AI", "About"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
        assert not app.error
        if page == "Explainable AI":
            app.selectbox[0].select_index(4).run()
            assert not app.exception
            assert len(app.metric) == 3


def test_missing_artifacts_offer_generation_command(tmp_path):
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    source = source.replace("from src.utils import ROOT, LABELS",
                            f"from src.utils import LABELS\nfrom pathlib import Path\nROOT = Path({str(tmp_path)!r})")
    # Streamlit 1.50's from_string uses the Windows locale instead of UTF-8.
    isolated_app = tmp_path / "isolated_app.py"
    isolated_app.write_text(source, encoding="utf-8")
    app = AppTest.from_file(str(isolated_app), default_timeout=60).run()
    assert not app.exception
    assert "python scripts/run_all.py" in app.error[0].value
