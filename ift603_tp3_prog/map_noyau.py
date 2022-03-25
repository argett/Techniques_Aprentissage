# -*- coding: utf-8 -*-

###
#   Eliott THOMAS         —  21 164 874
#   Lilian FAVRE GARCIA   —  21 153 421
#   Tsiory Razafindramisa —  21 145 627
###

import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.model_selection import train_test_split
from itertools import product
from tqdm import tqdm


class MAPnoyau:
    def __init__(self, lamb=0.2, sigma_square=1.06, b=1.0, c=0.1, d=1.0, M=2, noyau='rbf'):
        """
        Classe effectuant de la segmentation de données 2D 2 classes à l'aide de la méthode à noyau.

        lamb: coefficiant de régularisation L2
        sigma_square: paramètre du noyau rbf
        b, d: paramètres du noyau sigmoidal
        M,c: paramètres du noyau polynomial
        noyau: rbf, lineaire, olynomial ou sigmoidal
        """
        self.lamb = lamb
        self.a = None
        self.sigma_square = sigma_square
        self.M = M
        self.c = c
        self.b = b
        self.d = d
        self.noyau = noyau
        self.x_train = None

    def entrainement(self, x_train, t_train):
        """
        Entraîne une méthode d'apprentissage à noyau de type Maximum a
        posteriori (MAP) avec un terme d'attache aux données de type
        "moindre carrés" et un terme de lissage quadratique (voir
        Eq.(1.67) et Eq.(6.2) du livre de Bishop).  La variable x_train
        contient les entrées (un tableau 2D Numpy, où la n-ième rangée
        correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
        où le n-ième élément correspond à la cible t_n).

        L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
        ou polynomial (spécifié par ''self.noyau'') et dont les parametres
        sont contenus dans les variables self.sigma_square, self.c, self.b, self.d
        et self.M et un poids de régularisation spécifié par ``self.lamb``.

        Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
        l'equation 6.8 du livre de Bishop et garder en mémoire les données
        d'apprentissage dans ``self.x_train``
        """
        self.x_train = x_train
        # Gram matrix
        K = None
        if self.noyau == "rbf":
            tempA = np.zeros(shape=(x_train.shape[0], x_train.shape[0]), dtype=float)

            for i, j in product(range(x_train.shape[0]), range(x_train.shape[0])):
                tempB = x_train[i] - x_train[j]
                tempA[i, j] = -(np.square(tempB[0]) + np.square(tempB[1])) / (2 * self.sigma_square)

            K = np.exp(tempA)

        elif self.noyau == "lineaire":
            K = x_train@x_train.T

        elif self.noyau == "sigmoidal":
            K = np.tanh(self.b * x_train@x_train.T + self.d)

        elif self.noyau == "polynomial":
            K = np.power(((x_train@x_train.T) + self.c), self.M)

        else:
            print("\nMauvais noyau entré comme paramètre")
            sys.exit(1)

        self.a = np.linalg.solve(K + (np.identity(x_train.shape[0]) * self.lamb), np.identity(x_train.shape[0]))@t_train

    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.

        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée. Elle doit utiliser le champs ``self.a`` afin de calculer
        la prédiction y(x) (équation 6.9).

        NOTE : Puisque nous utilisons cette classe pour faire de la
        classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
        sinon
        """
        k = None
        if self.noyau == "rbf":
            tempA = np.zeros(shape=(self.x_train.shape[0]), dtype=float)

            tempB = self.x_train[:] - x
            tempA[:] = -(np.square(tempB[:, 0]) + np.square(tempB[:, 1])) / (2 * self.sigma_square)

            k = np.exp(tempA)

        elif self.noyau == "lineaire":
            k = self.x_train@x

        elif self.noyau == "sigmoidal":
            k = np.tanh(self.b * (self.x_train@x) + self.d)

        else:
            k = np.power(((self.x_train@x) + self.c), self.M)

        y = k@self.a

        if y > 0.5:
            return 1
        else:
            return 0

    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
        return np.square(t - prediction)

    def validation_croisee(self, x_tab, t_tab):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
        ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
        ``self.lamb`` avec une validation croisée de type "k-fold" où k=10 avec les
        données contenues dans x_tab et t_tab.  Une fois les meilleurs hyperparamètres
        trouvés, le modèle est entraîné une dernière fois.

        SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6

        """

        meilleur_err = np.inf
        num_fold = 10

        if self.noyau == "rbf":
            meilleur_sigma = -1
            meilleur_lamb = -1
            for lambda_param, hyper in tqdm(product(np.linspace(0.000000001, 2, 10), np.linspace(0.000000001, 2, 10))):
                self.sigma_square = hyper
                self.lamb = lambda_param
                sum_error = 0

                for k in range(num_fold):  # K-fold validation
                    X_train, X_val, y_train, y_val = train_test_split(x_tab, t_tab, test_size=0.2,
                                                                      random_state=k, shuffle=True)
                    self.entrainement(X_train, y_train)
                    sum_err_locale = 0
                    for i in range(X_val.shape[0]):
                        y_hat = self.prediction(X_val[i])  # vecteur de prédiction
                        sum_err_locale += np.sum(self.erreur(y_val[i], y_hat))
                    sum_error += sum_err_locale

                avg_err_locale = sum_error / (num_fold*X_val.shape[0])  # Moyenne des erreurs sur le K-fold
                if(avg_err_locale < meilleur_err):
                    meilleur_err = avg_err_locale
                    meilleur_sigma = hyper
                    meilleur_lamb = lambda_param

            self.lamb = meilleur_lamb
            self.sigma_square = meilleur_sigma
            self.entrainement(x_tab, t_tab)

        elif self.noyau == "lineaire":
            # juste lambda à tester
            meilleur_lamb = -1

            for lambda_param in tqdm(np.linspace(0.000000001, 2, 10)):
                self.lamb = lambda_param
                sum_error = 0

                for k in range(num_fold):  # K-fold validation
                    X_train, X_val, y_train, y_val = train_test_split(x_tab, t_tab, test_size=0.2,
                                                                      random_state=k, shuffle=True)
                    self.entrainement(X_train, y_train)
                    sum_err_locale = 0
                    for i in range(X_val.shape[0]):
                        y_hat = self.prediction(X_val[i])  # vecteur de prédiction
                        sum_err_locale += self.erreur(y_val[i], y_hat)
                    sum_error += sum_err_locale

                avg_err_locale = sum_error/(num_fold)  # Moyenne des erreurs sur le K-fold
                if(avg_err_locale < meilleur_err):
                    meilleur_err = avg_err_locale
                    meilleur_lamb = lambda_param

            self.lamb = meilleur_lamb
            self.entrainement(x_tab, t_tab)

        elif self.noyau == "sigmoidal":
            meilleur_B = -1
            meilleur_D = -1
            meilleur_lamb = -1
            for lambda_param, hyperD, hyperB in tqdm(product(np.linspace(0.000000001, 2, 10),
                                                             np.linspace(0.00001, 0.01, 10),
                                                             np.linspace(0.00001, 0.01, 10))):
                self.b = hyperB
                self.d = hyperD
                self.lamb = lambda_param
                sum_error = 0

                for k in range(num_fold):  # K-fold validation
                    X_train, X_val, y_train, y_val = train_test_split(x_tab, t_tab, test_size=0.2,
                                                                      random_state=k, shuffle=True)
                    self.entrainement(X_train, y_train)
                    sum_err_locale = 0
                    for i in range(X_val.shape[0]):
                        y_hat = self.prediction(X_val[i])  # vecteur de prédiction
                        sum_err_locale += self.erreur(y_val[i], y_hat)
                    sum_error += sum_err_locale

                avg_err_locale = sum_error/(num_fold)  # Moyenne des erreurs sur le K-fold
                if(avg_err_locale < meilleur_err):
                    meilleur_err = avg_err_locale
                    meilleur_B = hyperB
                    meilleur_D = hyperD
                    meilleur_lamb = lambda_param

            self.b = meilleur_B
            self.d = meilleur_D
            self.lamb = meilleur_lamb
            self.entrainement(x_tab, t_tab)

        else:  # polynomial
            meilleur_M = -1
            meilleur_C = -1
            meilleur_lamb = -1

            for lambda_param, hyperM, hyperC in tqdm(product(np.linspace(0.000000001, 2, 10),
                                                             range(2, 7),
                                                             np.linspace(0, 5, 100))):
                self.M = hyperM
                self.c = hyperC
                sum_error = 0
                self.lamb = lambda_param

                for k in range(num_fold):  # K-fold validation
                    X_train, X_val, y_train, y_val = train_test_split(x_tab, t_tab, test_size=0.2,
                                                                      random_state=k, shuffle=True)
                    self.entrainement(X_train, y_train)
                    sum_err_locale = 0
                    for i in range(X_val.shape[0]):
                        y_hat = self.prediction(X_val[i])  # vecteur de prédiction
                        sum_err_locale += np.sum(self.erreur(y_val[i], y_hat))
                    sum_error += sum_err_locale

                avg_err_locale = sum_error/(num_fold*X_val.shape[0])  # Moyenne des erreurs sur le K-fold
                if(avg_err_locale < meilleur_err):
                    meilleur_err = avg_err_locale
                    meilleur_M = hyperM
                    meilleur_C = hyperC
                    meilleur_lamb = lambda_param

            self.M = meilleur_M
            self.c = meilleur_C
            self.lamb = meilleur_lamb
            self.entrainement(x_tab, t_tab)

    def affichage(self, x_tab, t_tab):

        # Affichage
        ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
        iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
        iX, iY = np.meshgrid(ix, iy)
        x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
        contour_out = np.array([self.prediction(x) for x in x_vis])
        contour_out = contour_out.reshape(iX.shape)

        plt.contourf(iX, iY, contour_out > 0.5)
        plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')
        plt.show()
