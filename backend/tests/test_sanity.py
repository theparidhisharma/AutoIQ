from backend.predict import predict_failure

def run_sanity_tests():
    samples = [
        "Machine overheating with abnormal vibration",
        "Power fluctuation caused sudden shutdown",
        "Sensor calibration issue affecting readings"
    ]

    for text in samples:
        output = predict_failure(text)
        print("INPUT:", text)
        print("OUTPUT:", output)
        print("-" * 50)

if __name__ == "__main__":
    run_sanity_tests()
