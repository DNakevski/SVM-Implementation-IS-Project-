package svmlearn;

public class SVM {
        /** Istreniran/Loadiran model */
        private Model model;
        private double C = 1;
        private double tol = 10e-3;
        private double tol2 = 10e-5;
        /** Broj na iteracii preku alpha bez promeni */
        private int maxpass = 10;
        
        private double Ei, Ej;
        private double ai_old, aj_old;
        private double L, H;
        /* ---------------------------------------- */

        public SVM() {
        }
        /**
         * Treniranje na SVM
         * @param istreniraj go trenirackiot set.
         */
        public void svmTrain(Problem train) {
                KernelParams p = new KernelParams();
                svmTrain(train, p, 0);
        }
        /**
         * Treniranje na SVM so specificirani kernel parametri i algoritam.
         */
        public void svmTrain(Problem train, KernelParams p, int alg) {
                switch (alg) {
                case 0:
                        simpleSMO(train, p);
                        break;
                case 1:
                        SMO(train, p);
                        break;
                }
        }

        private void simpleSMO(Problem train, KernelParams p) {
                int pass = 0;
                int alpha_change = 0;
                int i, j;
                double eta;
                //Inicijalizacija:
                model = new Model();
                model.alpha = new double [train.l];
                model.b = 0;
                model.params = p;
                model.x = train.x;
                model.y = train.y;
                model.l = train.l;
                model.n = train.n;
                //Glavna Iteracija
                while (pass < maxpass) {
                        if (alpha_change > 0)
                                System.out.print(".");
                        else
                                System.out.print("*");
                        alpha_change = 0;
                        for (i=0; i<train.l; i++) {
                                Ei = svmTestOne(train.x[i]) - train.y[i];
                                if ((train.y[i]*Ei<-tol && model.alpha[i]<C) || (train.y[i]*Ei>tol && model.alpha[i]>0)) {
                                        j = (int)Math.floor(Math.random()*(train.l-1));
                                        j = (j<i)?j:(j+1);
                                        Ej = svmTestOne(train.x[j]) - train.y[j];
                                        ai_old = model.alpha[i];
                                        aj_old = model.alpha[j];
                                        L = computeL(train.y[i], train.y[j]);
                                        H = computeH(train.y[i], train.y[j]);
                                        if (L == H) //sledno i
                                                continue;
                                        eta = 2*kernel(train.x[i],train.x[j])-kernel(train.x[i],train.x[i])-kernel(train.x[j],train.x[j]);
                                        if (eta >= 0) //sledno i
                                                continue;
                                        model.alpha[j] = aj_old - (train.y[j]*(Ei-Ej))/eta;
                                        if (model.alpha[j] > H)
                                                model.alpha[j] = H;
                                        else if (model.alpha[j] < L)
                                                model.alpha[j] = L;
                                        if (Math.abs(model.alpha[j]-aj_old) < tol2) //sledno i
                                                continue;
                                        model.alpha[i] = ai_old + train.y[i]*train.y[j]*(aj_old-model.alpha[j]);
                                        computeBias(model.alpha[i], model.alpha[j], train.y[i], train.y[j], 
                                                        kernel(train.x[i], train.x[i]), kernel(train.x[j], train.x[j]), 
                                                        kernel(train.x[i], train.x[j]));
                                        alpha_change++;
                                }
                        }
                        if (alpha_change == 0)
                                pass++;
                        else
                                pass = 0;
                }
                System.out.println();
        }
        /**
         * Go presmetuva L.
         * @param yi
         * @param yj
         * @return vrakja L.
         */
        private double computeL(int yi, int yj) {
                double L = 0;
                if (yi != yj) {
                        L = Math.max(0, -ai_old+aj_old);
                } else {
                        L = Math.max(0, ai_old+aj_old-C);
                }
                return L;
        }
        /**
         * Go presmetuva  H.
         * @param yi
         * @param yj
         * @return Vrakja H.
         */
        private double computeH(int yi, int yj) {
                double H = 0;
                if (yi != yj) {
                        H = Math.min(C, -ai_old+aj_old+C);
                } else {
                        H = Math.min(C, ai_old+aj_old);
                }
                return H;
        }

        private void computeBias(double ai, double aj, int yi, int yj, double kii, double kjj, double kij) {
                double b1 = model.b - Ei - yi*(ai-ai_old)*kii - yj*(aj-aj_old)*kij;
                double b2 = model.b - Ej - yi*(ai-ai_old)*kij - yj*(aj-aj_old)*kjj;
                if (0 < ai && ai<C)
                        model.b = b1;
                else if (0 < aj && aj < C)
                        model.b = b2;
                else
                        model.b = (b1+b2)/2;            
        }
        /**
         * SMO Algoritam.
         * @param set za treniranje.
         * @param p parametri na kernel.
         */
        private void SMO(Problem train, KernelParams p) {
                
        }
        /**
         * Testiraj celosen data set
         */
        public int [] svmTest(Problem test) {
                if (test == null) 
                        return null;
                int [] pred = new int[test.l];
                for (int i=0; i<test.l; i++) {
                        pred[i] = (svmTestOne(test.x[i])<0?-1:1);
                }
                return pred;
        }
        /**
         * testiraj eden primer
         */
        public double svmTestOne(FeatureNode [] x) {
                double f = 0;
                for (int i=0; i<model.l; i++) {
                        f += model.alpha[i]*model.y[i]*kernel(x, model.x[i]);
                }
                return f+model.b;
        }

        private double kernel(FeatureNode [] x, FeatureNode [] z) {
                double ret = 0;
                switch (model.params.kernel) {
                case 0: //user defined
                        break;
                case 1: //linear
                        ret = Kernel.kLinear(x, z);
                        break;
                case 2: //polynomial
                        ret = Kernel.kPoly(x, z, model.params.a, model.params.b, model.params.c);
                        break;
                case 3: //gaussian
                        ret = Kernel.kGaussian(x, z, model.params.a);
                        break;
                case 4: //tanh
                        ret = Kernel.kTanh(x, z, model.params.a, model.params.b);
                        break;
                }
                return ret;
        }
        public Model getModel() {
                return model;
        }
        public void setModel(Model m) {
                model = m;
        }
        public double getC() {
                return C;
        }
        public void setC(double C) {
                this.C = C;
        }
        public double getTolerance() {
                return tol;
        }
        public void setTolerance(double tol) {
                this.tol = tol; 
        }
        public double getTolerance2() {
                return tol2;
        }
        public void setTolerance2(double tol) {
                this.tol2 = tol;
        }
        public int getMaxPass() {
                return maxpass;
        }
        public void setMaxPass(int p) { 
                maxpass = p;
        }
        public void setParameters(double C, double tol, double tol2, int maxpass) {
                this.C = C;
                this.tol = tol;
                this.tol2 = tol2;
                this.maxpass = maxpass;
        }
}