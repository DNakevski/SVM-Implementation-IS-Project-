package svmlearn;

public class KernelParams {
        /**
         * Tip na kernel;
         */
        public int kernel = 1;
        /** Parametar a i sigma */
        protected double a;
        /** Parametar b */
        protected double b;
        /** Parametar c */
        protected double c;
        public KernelParams(int k, double a, double b, double c) {
                this.kernel = k;
                this.a = a;
                this.b = b;
                this.c = c;
        }
        public KernelParams() {
                this(1,1,1,1);
        }
}