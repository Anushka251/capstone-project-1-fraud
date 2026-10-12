import streamlit as st
import score

st.set_page_config(
    page_title="AVENOR | Fraud Detection",
    page_icon="🛡️",
    layout="centered",
)

@st.cache_resource
def load_model():
    score.init()
    return True

try:
    load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

st.title("🛡️ Online Fraud Detection")
st.caption("Machine Learning · Random Forest · Transaction Risk Analysis")
st.write("Enter the transaction details below to evaluate its fraud risk.")

with st.form("transaction_form"):
    st.subheader("Transaction details")

    transaction_type = st.selectbox(
        "Transaction type",
        ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
    )
    step = st.number_input("Transaction step", min_value=0, value=1)
    amount = st.number_input("Transaction amount", min_value=0.0, value=5000.0)
    oldbalanceOrg = st.number_input(
        "Sender balance before", min_value=0.0, value=5000.0
    )
    newbalanceOrig = st.number_input(
        "Sender balance after", min_value=0.0, value=0.0
    )
    oldbalanceDest = st.number_input(
        "Recipient balance before", min_value=0.0, value=1000.0
    )
    newbalanceDest = st.number_input(
        "Recipient balance after", min_value=0.0, value=6000.0
    )

    submitted = st.form_submit_button(
        "Predict fraud risk",
        type="primary",
        use_container_width=True,
    )

if submitted:
    transaction = {
        "step": int(step),
        "type": transaction_type,
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalanceOrg),
        "newbalanceOrig": float(newbalanceOrig),
        "oldbalanceDest": float(oldbalanceDest),
        "newbalanceDest": float(newbalanceDest),
    }

    try:
        result = score.run(transaction)

        if "error" in result:
            st.error(result["error"])
        else:
            prediction = result["predictions"][0]
            probability = float(prediction["fraud_probability"])

            st.subheader("Prediction result")
            st.metric("Fraud probability", f"{probability:.2%}")
            st.progress(min(1.0, max(0.0, probability)))

            if prediction["label"] == "fraud":
                st.error("⚠️ Potentially fraudulent transaction")
            else:
                st.success("✅ Classified as legitimate")

            st.caption(
                "This is a model prediction, not a guarantee that a transaction is safe."
            )

            with st.expander("View prediction details"):
                st.json(prediction)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
