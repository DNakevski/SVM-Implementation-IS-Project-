package svmlearn;

public class Kernel {
        /**
         * Kalkulira kvadratna Evklidova odalecenost na 2 vektora
         * @param x prva tocka (vector)
         * @param z vtora tocka (vector)
         * @return kvadratna Evklidova odalecenost
         */
        public static double euclidean_dist2(FeatureNode [] x, FeatureNode [] z) {
                double sum=0;
                int i,j;
                for (i=0,j=0;x!=null && z!=null && i<x.length && j<z.length;) {
                        if (x[i].index<z[j].index) {
                                sum+=x[i].value*x[i].value;
                                i++;
                        }
                        else if (z[j].index<x[i].index) {
                                sum+=z[j].value*z[j].value;
                                j++;
                        }
                        else {
                                sum+=(x[i].value-z[j].value)*(x[i].value-z[j].value);
                                i++;
                                j++;
                        }
                }
                for (;x!=null && i<x.length;i++) {
                        sum+=x[i].value*x[i].value;
                }
                for (;z!=null && j<z.length;j++) {
                        sum+=z[j].value*z[j].value;
                }
                return sum;
        }
        /**
         * kalkulira dot proizvod na 2 vektora
         * @param x prviot vektor
         * @param z vtoriot vektor
         * @return dot proizvod na vektorite
         */
        public static double dot_product(FeatureNode [] x, FeatureNode [] z) {
                double sum=0;
                int i,j;
                for (i=0,j=0;x!=null && z!=null && i<x.length && j<z.length;) {
                        if (x[i].index<z[j].index) {
                                i++;
                        }
                        else if (z[j].index<x[i].index) {
                                j++;
                        }
                        else {
                                sum+=x[i].value*z[j].value;
                                i++;
                                j++;
                        }
                }
                return sum;
        }
        /**
         * Linearen kernel: k(x,z) = <x,z>
         * @param x prv vektor
         * @param z vtor vektor
         * @return linearna kernel vrednost
         */
        public static double kLinear(FeatureNode [] x, FeatureNode [] z) {
                return dot_product(x, z);
        }
        /**
         * Polinomijalen kernel: k(x,z) = (a*<x,z>+b)^c
         * @param x prv vektor
         * @param z vtor vektor
         * @param a koeficient na <x,z>
         * @param b bias
         * @param c power
         * @return polinomijalna kernel vrednost
         */
        public static double kPoly(FeatureNode [] x, FeatureNode [] z, double a, double b, double c) {
                if (c == 1.0)
                        return a*dot_product(x, z)+b;
                return Math.pow(a*dot_product(x, z)+b, c);
        }
        /**
         * Gausov (RBF) kernel: k(x,z) = (-0.5/sigma^2)*||x-z||^2
         * @param x prv vektor
         * @param z vtor vektor
         * @param sigma parametar (standardna devijacija)
         * @return Gausova kernel vrednost
         */
        public static double kGaussian(FeatureNode [] x, FeatureNode [] z, double sigma) {
                return (-0.5/sigma*sigma)*euclidean_dist2(x, z);
        }
        /**
         * Tanh-ov (sigmoid) kernel: k(x,z) = tanh(a*<x,z>+b)
         * @param x prv vektor
         * @param z vtor vektor
         * @param a koeficient na <x,z>
         * @param b bias
         * @return tanh kernel vrednost
         */
        public static double kTanh(FeatureNode [] x, FeatureNode [] z, double a, double b) {
                return Math.tanh(a*dot_product(x, z)+b);
        }
}