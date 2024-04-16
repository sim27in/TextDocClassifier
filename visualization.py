import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def generate_bar_graph(class_labels, probabilities, output_image_file):
    y_positions = range(len(class_labels))
    plt.figure(figsize=(9,5))
    bars = plt.barh(y_positions, probabilities, align="center", alpha=0.5, color="#2c8df2", edgecolor='black', linewidth=1.5)
    plt.yticks(y_positions, class_labels)
    plt.xlabel("Probabilities")
    plt.ylabel("Classes")
    plt.title("Class Probability Graph")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    for bar, prob in zip(bars, probabilities):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{prob:.2%}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_image_file)
    plt.close()