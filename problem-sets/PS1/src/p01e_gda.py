import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
 

    # *** START CODE HERE ***
     x , y = util.load_dataset(train_path,add_intercept=False)
    model = GDA()
    model.fit(x,y)
    x_equal , y_equal = util.loa_dataset(eval_path,add_intercept = False)
    model.predict(x_equal , y_equal)
    



    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
       m = x.shape[0]
    phi = np.mean(y)

    conteggio_y_1 = 0
    conteggio_y_0 = 0
    somma_x_0 = np.zeros(x.shape[1])
    somma_x_1 = np.zeros(x.shape[1])

    for i in range(len(y)):
        if y[i] == 1:
            somma_x_1 += x[i]
            conteggio_y_1 += 1
        else:
            somma_x_0 += x[i]
            conteggio_y_0 += 1

    mu_0 = somma_x_0 / conteggio_y_0
    mu_1 = somma_x_1 / conteggio_y_1

    prod = np.zeros((x.shape[1], x.shape[1]))

    for i in range(len(y)):
        if y[i] == 1:
            mu_corrente = mu_1
        else:
            mu_corrente = mu_0

        # x[i] e mu_corrente hanno shape (n,).
        # La loro differenza è quindi ancora un array monodimensionale (n,).
        # Su un array 1D, .T non cambia la forma.
        #
        # reshape(-1, 1) trasforma il vettore:
        #
        # [2, 3]          shape (2,)
        #
        # in:
        #
        # [[2],
        #  [3]]           shape (2, 1)
        #
        # -1 significa che NumPy calcola automaticamente il numero di righe.
        # 1 significa che vogliamo una sola colonna.
        #
        # In questo modo:
        # differenza.shape   = (n, 1)
        # differenza.T.shape = (1, n)
        #
        # e il prodotto:
        # (n, 1) @ (1, n)
        # restituisce una matrice (n, n).

        differenza = (x[i] - mu_corrente).reshape(-1, 1)
        prod += differenza @ differenza.T

    sigma = prod / m

    sigma_inv = np.linalg.inv(sigma)

    theta = sigma_inv @ (mu_1 - mu_0)

    theta_0 = (
        0.5
        * (mu_0 + mu_1).reshape(-1, 1).T
        @ sigma_inv
        @ (mu_0 - mu_1).reshape(-1, 1)
        + np.log(phi / (1 - phi))
    ).item()

    self.theta = np.concatenate(([theta_0], theta))

    
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a
        prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        theta_0 = self.theta[0]
        theta = self.theta[1:]
        logit = x @  theta + theta_0
        probabilities = 1/(1+ np.exp(-logit))
        return predictions = (probabilities > 0).astype(int) 
  
         
        # *** END CODE HERE
