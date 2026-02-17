# Brain-Tumor-Detection-using-Knowledge-Distillation
Ongoing Project
#  📌 Overview

Brain tumor detection from MRI images is a critical task in medical image analysis. Deep learning models achieve high accuracy but are often computationally expensive, limiting deployment in resource-constrained clinical environments.

This project implements Knowledge Distillation (KD) using a custom Distiller class in TensorFlow/Keras to transfer knowledge from a high-capacity teacher model to a lightweight student model, enabling efficient and accurate brain tumor classification.

# 🎯 Objective

Improve student model performance using teacher supervision

Reduce computational complexity for real-world deployment

Maintain high diagnostic accuracy

The dataset contains two classes:

Brain Tumor

Eye (Non-tumor class)

#  🧠 Knowledge Distillation Framework

Knowledge Distillation trains a smaller student model to mimic the soft probability outputs of a larger teacher model.


​
## 📌 Loss Function

The total loss is defined as:

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{student} + (1 - \alpha) \cdot \mathcal{L}_{distillation}
$$


Where:

Student Loss → Categorical Crossentropy

Distillation Loss → KL Divergence

Temperature (T = 5.0) → Softens probability distribution

Alpha (α = 0.5) → Balances student and distillation loss

#  🏗️ Distiller Model Architecture

The custom Distiller class extends keras.Model and includes:

🔹 Core Components

Teacher Model (frozen during training)

Student Model (trainable)

Temperature scaling

Alpha weighting factor

🔹 Metrics Tracked

Accuracy

Precision

Recall

AUC

Student Loss

Distillation Loss

Total Loss

#  ⚙️ Implementation Details
Training Flow

Forward pass through teacher (no gradient update)

Forward pass through student

Compute:

Hard label loss (Categorical Crossentropy)

Soft label loss (KL Divergence with temperature scaling)

Combine losses

Backpropagation updates only student model

Testing Flow

Only student model is evaluated

Distillation loss is not used during testing

#  📊 Evaluation Metrics

Accuracy – Overall classification performance

Precision – Tumor prediction correctness

Recall (Sensitivity) – Tumor detection capability

AUC – Discriminative ability

#  🚀 How to Use
1️⃣ Initialize Teacher and Student Models
teacher = build_teacher_model()
student = build_student_model()

2️⃣ Create Distiller
distiller = Distiller(
    student=student,
    teacher=teacher,
    temperature=5.0,
    alpha=0.5
)

3️⃣ Compile
distiller.compile(
    optimizer=tf.keras.optimizers.Adam()
)

4️⃣ Train
distiller.fit(train_dataset, validation_data=val_dataset, epochs=50)



# 🔬 Key Contributions

Custom Keras-based Knowledge Distillation pipeline

Balanced hard and soft label supervision

Improved lightweight model performance

Suitable for clinical edge deployment

# 🏥 Applications

Automated brain tumor screening

Clinical decision support systems

Medical AI edge deployment

Resource-limited healthcare setups

#  📈 Future Improvements

Multi-class tumor classification

Integration with Grad-CAM for explainability

Hybrid CNN-Transformer teacher models

Quantization-aware training for further compression

# 📜 Citation

If you use this work, please cite:

@article{brain_tumor_kd2026,
  title={Brain Tumor Detection using Knowledge Distillation},
  author={Your Name},
  year={2026}
}



Help write the methodology section for journal submission

Optimize the Distiller class for binary classification (since you have 2 classes)
