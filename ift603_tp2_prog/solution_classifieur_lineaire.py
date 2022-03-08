# -*- coding: utf-8 -*-

#####
#  Eliott THOMAS — 21 164 874
#  Lilian FAVRE GARCIA — 21 153 421
#  Tsiory Razafindramisa — 21 145 627
###

import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from itertools import product


class ClassifieurLineaire:
    def __init__(self, lamb, methode):
        """
        Algorithmes de classification lineaire

        L'argument ``lamb`` est une constante pour régulariser la magnitude
        des poids w et w_0

        ``methode`` :   1 pour classification generative
                        2 pour Perceptron
                        3 pour Perceptron sklearn
        """
        self.w = np.array([1., 2.])  # paramètre aléatoire
        self.w_0 = -5.              # paramètre aléatoire
        self.lamb = lamb
        self.methode = methode

    def entrainement(self, x_train, t_train):
        """
        Entraîne deux classifieurs sur l'ensemble d'entraînement formé des
        entrées ``x_train`` (un tableau 2D Numpy) et des étiquettes de classe cibles
        ``t_train`` (un tableau 1D Numpy).

        Lorsque self.method = 1 : implémenter la classification générative de
        la section 4.2.2 du libre de Bishop. Cette méthode doit calculer les
        variables suivantes:

        - ``p`` scalaire spécifié à l'équation 4.73 du livre de Bishop.

        - ``mu_1`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.75 du livre de Bishop.

        - ``mu_2`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.76 du livre de Bishop.

        - ``sigma`` matrice de covariance (tableau Numpy 2D) de taille DxD,
                    telle que spécifiée à l'équation 4.78 du livre de Bishop,
                    mais à laquelle ``self.lamb`` doit être ADDITIONNÉ À LA
                    DIAGONALE (comme à l'équation 3.28).

        - ``self.w`` un vecteur (tableau Numpy 1D) de taille D tel que
                    spécifié à l'équation 4.66 du livre de Bishop.

        - ``self.w_0`` un scalaire, tel que spécifié à l'équation 4.67
                    du livre de Bishop.

        lorsque method = 2 : Implementer l'algorithme de descente de gradient
                        stochastique du perceptron avec 1000 iterations

        lorsque method = 3 : utiliser la librairie sklearn pour effectuer une
                        classification binaire à l'aide du perceptron

        """
        if self.methode == 1:  # Classification generative
            print('Classification generative')

            N1 = np.count_nonzero(t_train == 1)
            N2 = np.count_nonzero(t_train == 0)

            p = N1/(N1+N2)

            elementsC1 = x_train[np.where(t_train == 1)]
            elementsC2 = x_train[np.where(t_train == 0)]

            # Moyennes
            mu_1 = (1/N1) * np.sum(elementsC1, axis=0)
            mu_2 = (1/N2) * np.sum(elementsC2, axis=0)

            cols1 = [np.reshape(elementsC1[k] - mu_1, (-1, 1)) for k in range(len(elementsC1))]
            cols2 = [np.reshape(elementsC2[k] - mu_2, (-1, 1)) for k in range(len(elementsC2))]

            intermediate1 = np.array([np.dot(cols1[k], cols1[k].T) for k in range(len(cols1))])
            intermediate2 = np.array([np.dot(cols2[k], cols2[k].T) for k in range(len(cols2))])

            S1 = (1/N1) * np.sum(intermediate1, axis=0)
            S2 = (1/N2) * np.sum(intermediate2, axis=0)

            S = p*S1 + (1-p)*S2
            S += np.identity(intermediate1.shape[1]) * self.lamb  # ne pas oublier la diagonale

            inv_S = np.linalg.solve(S, np.eye((S).shape[0]))

            self.w = inv_S.dot(mu_1 - mu_2)
            self.w_0 = -0.5*(mu_1.T)@inv_S@mu_1 + 0.5*(mu_2.T)@inv_S@mu_2 + np.log(N1/N2)

        elif self.methode == 2:  # Perceptron + SGD, learning rate = 0.001, nb_iterations_max = 1000
            print('Perceptron')
            eta = 0.001
            for iteration, (x_i, t_i) in product(range(1000), zip(x_train, t_train)):
                score = self.prediction(x_i)
                error = t_i - score
                self.w_0 += error * eta
                self.w += error * eta * x_i

        else:  # Perceptron + SGD [sklearn] + learning rate = 0.001 + penalty 'l2' voir http://scikit-learn.org/
            print('Perceptron [sklearn]')
            clf = Perceptron(tol=1e-3, random_state=42, penalty='l2')
            clf.fit(x_train, t_train, coef_init=self.w, intercept_init=self.w_0)
            w = clf.coef_
            w0 = clf.intercept_
            self.w = w[0]
            self.w_0 = w0[0]

        print('w = ', self.w, 'w_0 = ', self.w_0, '\n')

    def prediction(self, x):
        """
        Retourne la prédiction du classifieur lineaire.  Retourne 1 si x est
        devant la frontière de décision et 0 sinon.

        ``x`` est un tableau 1D Numpy

        Cette méthode suppose que la méthode ``entrainement()``
        a préalablement été appelée. Elle doit utiliser les champs ``self.w``
        et ``self.w_0`` afin de faire cette classification.
        """
        xT = x.reshape(-1, 1)
        score = np.matmul(self.w, xT) + self.w_0

        if score > 0:
            return 1
        else:
            return 0

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de classification, i.e.
        1. si la cible ``t`` et la prédiction ``prediction``
        sont différentes, 0. sinon.
        """
        # Cette fonction d'erreur est appelée nulle part
        # donc on n'a pas réellement pu tester la véracité des résultats
        # Nous en avons parlé avec Martin Valière et ce dernier nous a dit
        # qu'il était au courant de ce problème
        if t == prediction:
            return 1
        return 0

    def afficher_donnees_et_modele(self, x_train, t_train, x_test, t_test):
        """
        afficher les donnees et le modele

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=t_train * 100 + 20, c=t_train)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=t_test * 100 + 20, c=t_test)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')

        plt.show()

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.w_0, self.w
