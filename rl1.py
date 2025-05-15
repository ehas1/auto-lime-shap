# RLProject.py
import os
import sys
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

# Install required packages
def install_packages():
    packages = [
        "stable-baselines3[extra]",
        "'shimmy>=2.0'",
        "gymnasium",
        "lime",
        "shap"
    ]
    
    for package in packages:
        try:
            st.write(f"Installing {package}...")
            os.system(f"pip install {package}")
        except:
            st.error(f"Failed to install {package}. Please install it manually.")

# Check if running in Streamlit
if 'streamlit' in sys.modules:
    # Add a button to install dependencies
    if st.sidebar.button("Install Dependencies"):
        install_packages()

try:
    from stable_baselines3 import PPO
    import gymnasium as gym
    from lime.lime_tabular import LimeTabularExplainer
    import shap
    dependencies_loaded = True
except ImportError as e:
    st.error(f"Missing dependency: {str(e)}")
    st.info("Click 'Install Dependencies' in the sidebar to install required packages.")
    dependencies_loaded = False

# Only run the full app if dependencies are loaded
if dependencies_loaded:
    # Explainer utilities
    def load_model(path="models/cartpole_ppo"):
        try:
            return PPO.load(path)
        except FileNotFoundError:
            st.error(f"Model file not found at {path}. Please train a model first.")
            return None

    def predict(states: np.ndarray) -> np.ndarray:
        """
        Predicts actions for a batch of states and returns one-hot action probabilities.
        """
        model = load_model()
        if model is None:
            # Return dummy data if model is not available
            return np.zeros((len(states), 2))
        
        actions, _ = model.predict(states, deterministic=True)
        # One-hot encode the actions (assuming binary action space for CartPole)
        probs = np.zeros((len(actions), 2))  # 2 actions for CartPole: left (0) and right (1)
        for i, action in enumerate(actions):
            probs[i, action] = 1
        return probs

    def get_background_states(n_samples=100):
        env = gym.make("CartPole-v1")
        return np.array([env.observation_space.sample() for _ in range(n_samples)])

    def get_lime_explainer():
        bg = get_background_states()
        feature_names = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]
        class_names = ["left", "right"]
        return LimeTabularExplainer(
            training_data=bg,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True
        )

    def get_shap_explainer():
        bg = get_background_states()
        return shap.KernelExplainer(predict, bg)

    # Training functions
    def train_and_save(total_timesteps=100_000, save_path="models/cartpole_ppo"):
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=total_timesteps)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        st.success(f"Model saved to {save_path}")

    # Streamlit app
    def run_app():
        st.title("RL Policy Explainer: CartPole")
        
        tab1, tab2 = st.tabs(["Explain Model", "Train Model"])
        
        with tab1:
            # Sidebar: state sliders
            st.sidebar.subheader("State Parameters")
            labels = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
            ranges = [(-4.8, 4.8), (-5.0, 5.0), (-0.418, 0.418), (-5.0, 5.0)]
            state = np.array([
                st.sidebar.slider(label, float(low), float(high), float((low+high)/2))
                for label, (low, high) in zip(labels, ranges)
            ]).reshape(1, -1)

            # Explanation method
            method = st.sidebar.radio("Explain with", ["LIME", "SHAP"])

            if st.sidebar.button("Explain"):
                try:
                    if method == "LIME":
                        explainer = get_lime_explainer()
                        exp = explainer.explain_instance(state[0], predict, num_features=4, top_labels=1)
                        fig = exp.as_pyplot_figure(label=exp.available_labels()[0])
                    else:
                        explainer = get_shap_explainer()
                        shap_vals = explainer.shap_values(state)
                        fig, ax = plt.subplots()
                        feature_names = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]
                        ax.bar(feature_names, shap_vals[0])
                        ax.set_ylabel("SHAP value")
                        ax.set_title("Feature attributions (SHAP)")
                    
                    st.pyplot(fig)

                    # Download plot
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    st.download_button("Download Explanation", buf, file_name="explanation.png", mime="image/png")
                except Exception as e:
                    st.error(f"Error generating explanation: {str(e)}")
        
        with tab2:
            st.subheader("Train a new model")
            timesteps = st.number_input("Training timesteps", min_value=1000, max_value=1000000, value=100000, step=10000)
            model_path = st.text_input("Model save path", value="models/cartpole_ppo")
            
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    train_and_save(total_timesteps=timesteps, save_path=model_path)
else:
    st.title("RL Policy Explainer: CartPole")
    st.warning("Please install dependencies to continue")

# Make sure this is a proper Streamlit app
if __name__ == "__main__":
    if 'streamlit' in sys.modules:
        # In Streamlit, the app is already running
        if dependencies_loaded:
            run_app()
    else:
        # Command line execution
        print("To run the Streamlit app, use: streamlit run rl1.py")
        print("To train a model without Streamlit, uncomment the line below:")
        # train_and_save()
