import cv2
import os
import sqlite3
import numpy as np

# Caminho para o banco de dados SQLite
DB_PATH = "fingerprint_database.db"

# Configuração do banco de dados
def setup_database():
    """Configura o banco de dados SQLite para armazenar as minúcias e imagens."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS minutiae (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT UNIQUE NOT NULL,
            minutiae TEXT NOT NULL,
            image_blob BLOB
        )
    """)
    conn.commit()
    conn.close()

# Pré-processamento da imagem
def preprocess_image(file):
    """Executa o pré-processamento de uma imagem."""
    images = {}
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Imagem não encontrada: {file}")
    
    images['grayscale'] = image
    image = cv2.equalizeHist(image)
    images['histogram'] = image
    image = cv2.GaussianBlur(image, (1, 1), 0)
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    images['binary'] = binary_image
    return images

# Extração de minúcias
def extract_minutiae(binary_image):
    """Extrai as minúcias de uma imagem binária."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minutiae = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filtrar contornos pequenos
            for point in contour:
                x, y = point[0]
                minutiae.append((int(x), int(y)))  # Converter para tipo int
    return minutiae

# Armazenar minúcias e imagem no banco de dados
def store_data_in_db(image_name, minutiae, image):
    """Armazena as minúcias e a imagem no banco de dados."""
    minutiae = [(int(x), int(y)) for x, y in minutiae]
    _, image_encoded = cv2.imencode('.png', image)
    image_blob = image_encoded.tobytes()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO minutiae (image_name, minutiae, image_blob) 
        VALUES (?, ?, ?)""", 
        (image_name, str(minutiae), image_blob))
    conn.commit()
    conn.close()

# Recuperar dados do banco de dados
def get_all_data_from_db():
    """Recupera todas as entradas da tabela de minúcias."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, image_name, minutiae, image_blob FROM minutiae
    """)
    results = cursor.fetchall()
    conn.close()
    
    data = []
    for result in results:
        minutiae = eval(result[2])  # Converter string de volta para lista
        image_blob = np.frombuffer(result[3], dtype=np.uint8)
        image = cv2.imdecode(image_blob, cv2.IMREAD_GRAYSCALE)
        data.append((result[0], result[1], minutiae, image))
    return data

# Comparação de impressões digitais
def match_fingerprints(minutiae1, minutiae2):
    """Compara duas listas de minúcias e retorna uma pontuação de similaridade."""
    match_score = abs(len(minutiae1) - len(minutiae2))
    return match_score

# Funções para exibição no terminal
def print_green(text):
    print(f"\033[92m{text}\033[0m")  # ANSI: Verde

def print_white(text):
    print(f"\033[97m{text}\033[0m")  # ANSI: Branco

# Fluxo principal
if __name__ == "__main__":
    # Configurar o banco de dados
    setup_database()
    dataset_dir = "fingerprint_dataset/"
    reference_image_name = "102_5.tif"
    reference_image_path = os.path.join(dataset_dir, reference_image_name)
    
    try:
        print("\nPROCESSANDO A IMAGEM DE REFERÊNCIA...\n")
        ref_images = preprocess_image(reference_image_path)
        ref_minutiae = extract_minutiae(ref_images['binary'])
        
        # Armazenar minúcias e imagem de referência no banco de dados
        store_data_in_db(reference_image_name, ref_minutiae, ref_images['grayscale'])
        
        print("\nINICIANDO A COMPARAÇÃO DA IMAGEM DE REFERÊNCIA COM O BANCO DE DADOS...\n")
        threshold = 100
        all_data = get_all_data_from_db()
        
        for entry_id, image_name, db_minutiae, db_image in all_data:
            if image_name != reference_image_name:  # Não comparar consigo mesmo
                match_score = match_fingerprints(ref_minutiae, db_minutiae)
                
                # Exibir resultados
                if match_score < threshold:
                    print_green(f"ID {entry_id} - {image_name}: Impressões digitais coincidem! Pontuação: {match_score}")
                    cv2.imshow(f"Coincide - {image_name}", db_image)
                else:
                    print_white(f"ID {entry_id} - {image_name}: Impressões digitais não coincidem. Pontuação: {match_score}")
        
        # Aguarda até o usuário pressionar uma tecla para fechar as janelas
        print("\nPressione qualquer tecla para fechar as imagens...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Erro: {e}")
