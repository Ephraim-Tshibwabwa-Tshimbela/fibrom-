from PIL import Image
import os

def combine_results_images(results_dir="results/"):
    """Combine les graphiques de résultats en une seule image avec Pillow"""
    img1_path = os.path.join(results_dir, "training_history.png")
    img2_path = os.path.join(results_dir, "confusion_matrix_roc.png")
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("Images manquantes pour la combinaison.")
        return

    # Ouvrir les images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Redimensionner pour avoir la même largeur (basé sur img1)
    target_width = 1200
    ratio1 = target_width / img1.width
    ratio2 = target_width / img2.width
    
    img1_new = img1.resize((target_width, int(img1.height * ratio1)))
    img2_new = img2.resize((target_width, int(img2.height * ratio2)))
    
    # Créer une nouvelle image
    total_height = img1_new.height + img2_new.height
    combined_img = Image.new('RGB', (target_width, total_height), (255, 255, 255))
    
    # Coller les images
    combined_img.paste(img1_new, (0, 0))
    combined_img.paste(img2_new, (0, img1_new.height))
    
    # Sauvegarder
    save_path = os.path.join(results_dir, "rapport_visuel_global.png")
    combined_img.save(save_path)
    print(f"✅ Image combinée générée avec Pillow: {save_path}")

if __name__ == "__main__":
    combine_results_images()
