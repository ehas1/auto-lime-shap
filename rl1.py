# RLProject.py
import os
import streamlit as st
import numpy as np
import gym
from stable_baselines3 import PPO
from lime.lime_tabular import LimeTabularExplainer
import shap
import matplotlib.pyplot as plt
import io

# ------------ Model & Explainer Utilities ------------
@st.cache_resource

def load_model(path: str = "models/cartpole_ppo") -> PPO:
    """
    Load a trained PPO model from the given path (without .zip extension).
    """
    model_path = path if path.endswith(".zip") else f"{path}.zip"
    if not os.path.exists(model_path):
        return None
    return PPO.load(path)


def predict(states: np.ndarray) -> np.ndarray:
    """
    Predicts actions for a batch of states and returns one-hot action probabilities.
    """
    model = load_model()
    if model is None:
        # If no model is loaded, return zeros to avoid errors
        return np.zeros((len(states), 2))
    actions, _ = model.predict(states, deterministic=True)
    probs = np.zeros((len(actions), model.action_space.n))
    for i, action in enumerate(actions):
        probs[i, action] = 1
    return probs

@st.cache_resource

def get_background_states(n_samples: int = 100) -> np.ndarray:
    """
    Sample random states from the CartPole environment for explainers.
    """
    env = gym.make("CartPole-v1")
    return np.array([env.observation_space.sample() for _ in range(n_samples)])

@st.cache_resource

def get_lime_explainer() -> LimeTabularExplainer:
    """
    Initialize a LIME explainer with background states.
    """
    bg = get_background_states()
    return LimeTabularExplainer(
        training_data=bg,
        feature_names=["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"],
        class_names=["left", "right"],
        discretize_continuous=True
    )

@st.cache_resource

def get_shap_explainer():
    """
    Initialize a SHAP explainer with background states.
    """
    bg = get_background_states()
    return shap.KernelExplainer(predict, bg)

# ------------ Training Function ------------
def train_and_save(total_timesteps: int = 100_000, save_path: str = "models/cartpole_ppo") -> None:
    """
    Train a PPO agent on CartPole-v1 and save the model.
    """
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    st.success(f"Model saved to {save_path}")

# ------------ Streamlit App ------------
def run_app():
    st.title("RL Policy Explainer: CartPole")
    tab1, tab2 = st.tabs(["Explain Model", "Train Model"])

    with tab1:
        st.subheader("Explain a Saved Policy")
        # Sidebar for state inputs
        labels = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
        ranges = [(-4.8, 4.8), (-5.0, 5.0), (-0.418, 0.418), (-5.0, 5.0)]
        state = np.array([
            st.sidebar.slider(label, float(low), float(high), float((low + high) / 2))
            for label, (low, high) in zip(labels, ranges)
        ]).reshape(1, -1)

        method = st.sidebar.radio("Choose Explainer", ["LIME", "SHAP"])
        if st.sidebar.button("Generate Explanation"):
            try:
                if method == "LIME":
                    explainer = get_lime_explainer()
                    exp = explainer.explain_instance(state[0], predict, num_features=4, top_labels=1)
                    fig = exp.as_pyplot_figure(label=exp.available_labels()[0])
                else:
                    explainer = get_shap_explainer()
                    shap_vals = explainer.shap_values(state)
                    fig, ax = plt.subplots()
                    ax.bar(["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"], shap_vals[0])
                    ax.set_ylabel("SHAP value")
                    ax.set_title("Feature attributions (SHAP)")
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                st.download_button("Download Explanation", buf, file_name="explanation.png", mime="image/png")
            except Exception as e:
                st.error(f"Error generating explanation: {e}")

    with tab2:
        st.subheader("Train a New Policy")
        timesteps = st.number_input("Number of Training Timesteps", min_value=1_000, max_value=1_000_000, value=100_000, step=10_000)
        model_path = st.text_input("Model Save Path", value="models/cartpole_ppo")
        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                train_and_save(total_timesteps=timesteps, save_path=model_path)

if __name__ == "__main__":
    run_app()
