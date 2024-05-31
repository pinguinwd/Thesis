from MNUECA_Lyapunov import *

if __name__ == '__main__':
    STEPS = 100
    SIZE = 101
    file_path = r'D:/PythonProjects/Thesis-data/smallCA/Data/'
    lyapnov_matrix, kolmogorov_matrix, shannon_matrix = generate_matrices(file_path, STEPS, SIZE)
    save_data(lyapnov_matrix, 'lyapnov_matrix')
    save_data(kolmogorov_matrix, 'kolmogorov_matrix')
    save_data(shannon_matrix, 'shannon_matrix')