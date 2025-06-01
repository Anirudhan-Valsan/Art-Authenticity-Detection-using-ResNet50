from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
from skimage.filters import sobel

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB limit

# Load model with custom objects
model = load_model('art_authenticity_model.keras')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_artifacts(img_array):
    """Enhanced artifact analysis with multiple detection heuristics"""
    # Convert to grayscale for some analyses
    gray = np.mean(img_array, axis=-1)
    
    # Calculate basic statistics
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Edge analysis 
    edges = sobel(gray)
    edge_intensity = np.mean(edges)
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'edge_intensity': edge_intensity
    }

def predict_image(image_path):
    try:
        # Load and preprocess image 
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        
        
        # Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)
        
        # 2. Normalization based on ImageNet standards
        img_array = img_array / 255.0  
        
        # Make prediction 
        prediction = model.predict(img_array)
        
        
        ai_prob = float(prediction[0][0])  
        human_prob = 1 - ai_prob
        
        # Adjust threshold based on validation metrics
        threshold = 0.5  
        
        #artifact analysis
        artifacts = analyze_artifacts(img_array[0])
        
        # Create visualization
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(2, 3)
        
        # Original image
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Uploaded Image', pad=10)
        
        # Confidence plot - fixed labels to match model interpretation
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.barh(['Human', 'AI'], [human_prob, ai_prob], 
                       color=['#4CAF50', '#F44336'])
        ax2.axvline(x=threshold, color='#333', linestyle='--', 
                   linewidth=1, label=f'Threshold ({threshold*100:.0f}%)')
        ax2.set_xlim(0, 1)
        ax2.set_xticks(np.arange(0, 1.1, 0.2))
        ax2.set_title('Authenticity Confidence', pad=10, fontsize=12)
        ax2.grid(axis='x', linestyle=':', alpha=0.5)
        
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{width*100:.1f}%', ha='left', va='center')
        
        # Artifact analysis
        ax3 = fig.add_subplot(gs[0, 2])
        artifact_metrics = [
            ('Brightness', artifacts['brightness']/255, '#FFA000'),
            ('Contrast', artifacts['contrast']/100, '#7CB342'),
            ('Edges', artifacts['edge_intensity']*10, '#5C6BC0')
        ]
        for i, (label, value, color) in enumerate(artifact_metrics):
            ax3.barh([i], [value], color=color)
            ax3.text(value + 0.02, i, f'{value:.2f}', ha='left', va='center')
        ax3.set_yticks(range(len(artifact_metrics)))
        ax3.set_yticklabels([m[0] for m in artifact_metrics])
        ax3.set_xlim(0, 1)
        ax3.set_title('Artifact Analysis', pad=10, fontsize=12)
        ax3.grid(axis='x', linestyle=':', alpha=0.5)
        
        # Decision explanation - fixed logic
        ax4 = fig.add_subplot(gs[1, 1:])
        ax4.axis('off')
        
        decision_text = []
        if ai_prob > threshold:
            decision_text.append(f"🔴 AI-GENERATED (Confidence: {ai_prob*100:.1f}%)")
            if ai_prob > 0.85:
                decision_text.append("- Very high confidence of AI generation")
            elif ai_prob > 0.7:
                decision_text.append("- Strong indicators of AI generation")
            else:
                decision_text.append("- Moderate confidence of AI generation")
        else:
            decision_text.append(f"🟢 HUMAN-CREATED (Confidence: {human_prob*100:.1f}%)")
            if human_prob > 0.85:
                decision_text.append("- Very high confidence of human creation")
            elif human_prob > 0.7:
                decision_text.append("- Strong indicators of human creation")
            else:
                decision_text.append("- Moderate confidence of human creation")
        
        # Add artifact insights
        if artifacts['edge_intensity'] < 0.05:
            decision_text.append("- Low edge intensity (common in AI images)")
        elif artifacts['edge_intensity'] > 0.15:
            decision_text.append("- High edge intensity (common in human art)")
            
        if artifacts['contrast'] < 25:
            decision_text.append("- Low contrast (common in AI images)")
            
        ax4.text(0.02, 0.8, "\n".join(decision_text), 
                ha='left', va='top', fontsize=11)
        
        plt.tight_layout()
        
        # Save plot to bytes
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches='tight', dpi=120)
        plt.close()
        img_bytes.seek(0)
        plot_url = base64.b64encode(img_bytes.getvalue()).decode('utf8')
        
        return {
            'ai_prob': ai_prob,
            'human_prob': human_prob,
            'threshold': threshold,
            'is_ai': ai_prob > threshold,
            'plot_url': plot_url,
            'artifacts': artifacts,
            'decision_text': decision_text
        }
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = secure_filename(file.filename)
            filename = f"{timestamp}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            result = predict_image(filepath)
            
            if result:
                return render_template('result.html',
                                    filename=filename,
                                    result=result,
                                    original_filename=original_filename)
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)