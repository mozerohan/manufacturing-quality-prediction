

from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["version"] == "1.0.0"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    payload = {
        "Rated_Power_VA": 500,
        "Input_Voltage_V": 230,
        "Winding_Time_Min": 45,
        "Assembly_Time_Min": 30,
        "Core_Weight_Kg": 2.5,
        "Copper_Weight_Kg": 1.2,
        "Winding_Temperature_C": 65,
        "Product_Type": "Control Transformer",
        "Production_Line": "Line A",
        "Shift": "Day",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in ["Pass", "Fail"]
    assert "Pass" in result["probabilities"]
    assert "Fail" in result["probabilities"]


def test_prediction_rejects_negative_winding_time():
    payload = {
        "Rated_Power_VA": 500,
        "Input_Voltage_V": 230,
        "Winding_Time_Min": -45,
        "Assembly_Time_Min": 30,
        "Core_Weight_Kg": 2.5,
        "Copper_Weight_Kg": 1.2,
        "Winding_Temperature_C": 65,
        "Product_Type": "Control Transformer",
        "Production_Line": "Line A",
        "Shift": "Day",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_rejects_missing_field():
    payload = {
        "Input_Voltage_V": 230,
        "Winding_Time_Min": 45,
        "Assembly_Time_Min": 30,
        "Core_Weight_Kg": 2.5,
        "Copper_Weight_Kg": 1.2,
        "Winding_Temperature_C": 65,
        "Product_Type": "Control Transformer",
        "Production_Line": "Line A",
        "Shift": "Day",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
