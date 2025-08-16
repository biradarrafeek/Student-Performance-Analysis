import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Generate marks (10 students × 5 subjects, marks between 30–100)
marks = np.random.randint(30, 101, size=(10, 5))

print("Marks Matrix (10 students × 5 subjects):\n", marks)

print("Shape:", marks.shape)   # (10, 5)
print("Dimensions:", marks.ndim) # 2
print("Data type:", marks.dtype)

# Marks of 1st student
print("Student 1 marks:", marks[0])

# Marks of all students in Subject 3
print("Subject 3 marks:", marks[:, 2])

# Marks of first 3 students
print("First 3 students:\n", marks[:3])

# Average marks of each student
student_avg = marks.mean(axis=1)
print("Student-wise average:", student_avg)

# Average marks of each subject
subject_avg = marks.mean(axis=0)
print("Subject-wise average:", subject_avg)

# Overall class average
print("Class Average:", marks.mean())

failed_students = np.any(marks < 40, axis=1)
print("Failed Students (True = Failed):", failed_students)

# List of failed students
print("Indexes of failed students:", np.where(failed_students)[0])

percentages = marks / 100
print("Percentages:\n", percentages)

total_marks = marks.sum(axis=1)
top3 = np.argsort(total_marks)[-3:][::-1]
print("Top 3 Students by total marks:", top3)

# Reshape into (5 subjects × 10 students) matrix
reshaped = marks.T
print("Reshaped Matrix (Subjects × Students):\n", reshaped)

# Weighted marks (different weight per subject)
weights = np.array([0.2, 0.25, 0.15, 0.2, 0.2])
weighted_scores = marks @ weights
print("Weighted Scores:", weighted_scores)
