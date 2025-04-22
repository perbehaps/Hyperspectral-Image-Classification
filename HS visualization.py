# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt  
import scipy.io
import requests
from pathlib import Path
import seaborn as sns
# from google.colab import drive  

class IndianPinesVisualizer:
    
    CLASS_NAMES = {
        0: 'Background',
        1: 'Alfalfa',
        2: 'Corn-notill',
        3: 'Corn-mintill',
        4: 'Corn',
        5: 'Grass-pasture',
        6: 'Grass-trees',
        7: 'Grass-pasture-mowed',
        8: 'Hay-windrowed',
        9: 'Oats',
        10: 'Soybean-notill',
        11: 'Soybean-mintill',
        12: 'Soybean-clean',
        13: 'Wheat',
        14: 'Woods',
        15: 'Buildings-Grass-Trees-Drives',
        16: 'Stone-Steel-Towers'
    }
    
    def __init__(self):
        self.data = None
        self.labels = None
        self.wavelengths = None

    def download_dataset(self):
            """
            Download Indian Pines dataset if not already present
            """
            # Creating data directory if it doesn't exist
            Path('data').mkdir(exist_ok=True)
            
            # URLs for the dataset
            data_url = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
            labels_url = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
            
            # Downloading data files if they don't exist
            for url, filename in [(data_url, 'indian_pines_corrected.mat'), 
                                (labels_url, 'indian_pines_gt.mat')]:
                file_path = Path('data') / filename
                if not file_path.exists():
                    print(f"Downloading {filename}...")
                    response = requests.get(url)
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded {filename}")

    def load_data(self):
        """
        Load the Indian Pines dataset
        """
        # Downloading the dataset if necessary
        # self.download_dataset()
        
        # Loading the data and ground truth
        data = scipy.io.loadmat('C:\\Users\Srivatsa\OneDrive\Desktop\Major Project\Hyperspectral-master\Hyperspectral-master\Data\Indian_pines_corrected.mat')
        labels = scipy.io.loadmat('C:\\Users\Srivatsa\OneDrive\Desktop\Major Project\Hyperspectral-master\Hyperspectral-master\Data\Indian_pines_gt.mat')
        
        self.data = data['indian_pines_corrected']
        self.labels = labels['indian_pines_gt']
        
        # Generating wavelengths (approximately 400-2500 nm)
        self.wavelengths = np.linspace(400, 2500, self.data.shape[2])
        
        print(f"Data shape: {self.data.shape}")
        print(f"Labels shape: {self.labels.shape}")

    def create_false_color(self, bands=[30, 20, 10]):
        """
        Creating false color composite from specified bands
        """
        false_color = np.dstack([self.data[:,:,b] for b in bands])
        # Normalize to 0-1 range
        false_color = (false_color - false_color.min()) / (false_color.max() - false_color.min())
        return false_color
    
    def plot_class_distribution(self):
        """
        Plotting distribution of classes in the dataset
        """
        if self.labels is None:
            raise ValueError("No data loaded")
            
        # Counting instances of each class
        unique, counts = np.unique(self.labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        # Creating a bar plot
        plt.figure(figsize=(15, 5))
        bars = plt.bar(
            [self.CLASS_NAMES[i] for i in unique],
            counts
        )
        plt.xticks(rotation=45, ha='right')
        plt.title('Distribution of Classes in Indian Pines Dataset')
        plt.xlabel('Class')
        plt.ylabel('Number of Pixels')
        
        # Adding value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom'
            )
        
        plt.tight_layout()
        plt.show()

    def plot_sample_spectra(self):
        """
        Plotting sample spectra from different classes
        """
        plt.figure(figsize=(15, 5))
            
        # Select a few major classes
        selected_classes = [2, 3, 10, 11, 14]  # Corn and Soybean varieties, Woods
            
        for class_id in selected_classes:
            # Get pixels belonging to this class
            mask = self.labels == class_id
            if np.any(mask):
                # Get mean spectrum for this class
                mean_spectrum = np.mean(self.data[mask], axis=0)
                plt.plot(self.wavelengths, mean_spectrum, label=self.CLASS_NAMES[class_id])
            
        plt.title('Average Spectra for Major Classes')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_comprehensive_view(self):
        """
        Create a comprehensive visualization of the dataset
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # False color composite
        false_color = self.create_false_color()
        axes[0, 0].imshow(false_color)
        axes[0, 0].set_title('False Color Composite')
           
        # Ground truth
        im = axes[0, 1].imshow(self.labels)
        axes[0, 1].set_title('Ground Truth Classes')
        plt.colorbar(im, ax=axes[0, 1])
            
        # Single band visualization
        mid_band = self.data.shape[2] // 2
        im = axes[1, 0].imshow(self.data[:, :, mid_band])
        axes[1, 0].set_title(f'Single Band ({int(self.wavelengths[mid_band])}nm)')
        plt.colorbar(im, ax=axes[1, 0])
            
        # Spectral variance
        spectral_variance = np.std(self.data, axis=2)
        im = axes[1, 1].imshow(spectral_variance)
        axes[1, 1].set_title('Spectral Variance')
        plt.colorbar(im, ax=axes[1, 1])
            
        plt.tight_layout()
        plt.show()

    def visualize_dataset(self):
        """
        Create all visualizations
        """
        print("Creating comprehensive visualization...")
        self.plot_comprehensive_view()
                
        print("\nPlotting class distribution...")
        self.plot_class_distribution()
               
        print("\nPlotting sample spectra...")
        self.plot_sample_spectra()
    
# Create visualizer instance
visualizer = IndianPinesVisualizer()
# Load the dataset
visualizer.load_data()
# Create all visualizations
visualizer.visualize_dataset()

# block diagram of the overall architecture
# not more than 4point in a slide
# add SVM and random forest 
# citation instead of link and add specific parameters in lit survey
# table containing Paper, Models used, Dataset used, accuracy, comments and citation
# CNN architecture diagram
# implement cross validation optimization
# add confusion matrix
# add TSNE plot