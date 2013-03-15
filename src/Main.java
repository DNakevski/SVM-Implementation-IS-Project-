import svmlearn.*;	

public class Main {
        public static void main(String [] args) {  
                SVM s = new SVM();
                
                Problem train = new Problem();
                Problem test = new Problem();
                
                //loadiraj go mnozestvoto za testiranje i mnozestvoto za treniranje
                train.loadBinaryProblem("D:\\train.txt");
                test.loadBinaryProblem("D:\\test.txt");

                System.out.println("Loaded.");
                System.out.println("Training...");
                s.svmTrain(train);
                System.out.println("Testing...");
                int [] pred = s.svmTest(test);
                for (int i=0; i<pred.length; i++)
                        System.out.println(pred[i]);
                
                EvalMeasures e = new EvalMeasures(test, pred, 2);
                System.out.println("Accuracy=" + e.Accuracy());
                
                System.out.println("Done.");
        }
}