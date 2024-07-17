from flask import Flask, render_template, request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns  # Import Seaborn for additional plots
import io
import base64

file = 'file/'
app = Flask(__name__, template_folder=file)

def generate_dot(seq1, seq2):
    len_seq1 = len(seq1)
    len_seq2 = len(seq2)
    dotmatrix = np.zeros((len_seq1, len_seq2))

    for i in range(len_seq1):
        for j in range(len_seq2):
            if seq1[i] == seq2[j]:
                dotmatrix[i][j] = 1

    plt.imshow(dotmatrix, cmap='Greys', interpolation='none', aspect='auto')
    plt.title('Dot Plot')
    plt.ylabel('Sequence 2')
    plt.xlabel('Sequence 1')
    plt.xticks(range(len_seq2), list(seq2))
    plt.yticks(range(len_seq1), list(seq1))

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return img_base64

def generate_histogram(seq1, seq2):
    combined_seq = seq1 + seq2
    characters = sorted(set(combined_seq))
    counts = [combined_seq.count(char) for char in characters]

    plt.bar(characters, counts, color='blue')
    plt.title('Character Frequency Histogram')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return img_base64

def generate_gc_content_plot(seq1, seq2):
    def gc_content(seq):
        return (seq.count('G') + seq.count('C')) / len(seq) * 100

    gc_contents = [gc_content(seq1), gc_content(seq2)]
    labels = ['Sequence 1', 'Sequence 2']

    plt.bar(labels, gc_contents, color='green')
    plt.title('GC Content')
    plt.ylabel('GC Content (%)')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return img_base64

def generate_violin_plot(seq1, seq2):
    # Example data for violin plot (replace with your own)
    data = np.random.randn(100, 2)

    violin = sns.violinplot(data=data)
    plt.title('Violin Plot')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return img_base64


def generate_heatmap(seq1, seq2):
    # Example data for heatmap (replace with your own)
    data = np.random.rand(10, 10)

    heatmap = sns.heatmap(data)
    plt.title('Heatmap')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return img_base64
def generate_scatter_plot(seq1, seq2):
    positions = list(range(len(seq1)))
    values1 = [ord(char) for char in seq1]
    values2 = [ord(char) for char in seq2]

    plt.scatter(positions, values1, alpha=0.5, label='Sequence 1')
    plt.scatter(positions, values2, alpha=0.5, label='Sequence 2')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.title('Scatter Plot of Sequences')
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    dot_plot = histogram = gc_content_plot = violin_plot  = heatmap = scatter_plot = None
    selected_graphs = []
    if request.method == 'POST':
        sequence1 = request.form['sequence1']
        sequence2 = request.form['sequence2']
        selected_graphs = request.form.getlist('graphs')

        if 'dot_plot' in selected_graphs:
            dot_plot = generate_dot(sequence1, sequence2)
        if 'histogram' in selected_graphs:
            histogram = generate_histogram(sequence1, sequence2)
        if 'gc_content_plot' in selected_graphs:
            gc_content_plot = generate_gc_content_plot(sequence1, sequence2)
        if 'violin_plot' in selected_graphs:
            violin_plot = generate_violin_plot(sequence1, sequence2)
        
        if 'heatmap' in selected_graphs:
            heatmap = generate_heatmap(sequence1, sequence2)
        
        if 'scatter_plot' in selected_graphs:
            scatter_plot = generate_scatter_plot(sequence1, sequence2)
        
    return render_template('index.html', dot_plot=dot_plot, histogram=histogram, gc_content_plot=gc_content_plot,
violin_plot=violin_plot, heatmap=heatmap,scatter_plot=scatter_plot,selected_graphs=selected_graphs)


if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True, port=5500)
