from flask import Flask, request, jsonify

app = Flask(__name__)

# Simple endpoint to receive predictions
@app.route("/api/predict", methods=["POST"])
def receive_prediction():
    data = request.json
    print("📩 Received from Edge:", data)  # log on server side

    # Here you can save to a file or database if you want
    return jsonify({"status": "success", "received": data})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
