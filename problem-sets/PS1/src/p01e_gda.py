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
        somma_x_1 =  np.zeros(x.shape[1])

        for i in range (len(y)):
            if y[i] == 1:
                somma_x_1 += x[i]
                conteggio_y_1 += 1
            else :
                 somma_x_0 += x[i]
                 conteggio_y_0 += 1

       mu_0 = somma_x_0/conteggio_y_0
       mu_1 = somma_x_1/conteggio_y_1
       prod = np.zeros(x.shape[1], x.shape[1]))
       for i range (len(y)):
          if y[i] == 1:
           mu_corrente = mu_1
           else:
           mu_corrente = mu_0
        #x.shape (m,n)
        #x[i] (n,)
        #mu.shape (n,)  / (x[i] - mu ) (n,)   / (x[i] - mu).T (,n)
          differenza = (x[i] - mu_corrente).reshape(-1, 1) 
   '''il rashape cambia forma perche il .T non funziona sui mono dimendionalei
    es    differenza = x[i] - mu_corrente =  [2, 3] La sua shape è: (2,)
    Questo è un array monodimensionale. NumPy non lo considera chiaramente né riga né colonna.
    con differenza.reshape(-1, 1) diventa : [
    [2],
    [3]
]
la shape quindi dievntra (2,1) cioè un vettore colonna.
Il -1 significa: calcola automaticamente quante righe servono.
Il 1 significa: voglio una sola colonna.
(n,)      array 1D
(n, 1)    vettore colonna
(1, n)    vettore riga
    '''      
          prod += differenza @ differenza.T  #(n,n)
 
 
    
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
        # *** END CODE HERE
