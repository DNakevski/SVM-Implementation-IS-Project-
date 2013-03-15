package svmlearn;

import java.io.*;
import java.util.*;

public class Problem {
        /** Broj na elementi na obucuvackoto mnozestvo */
        public int l;
        /** broj na karakteristiki */
        public int n;
        /** Array koj gi sodrzi celnite vrednosti */
        public int[] y;
        public CategoryMap<Integer> catmap;
        public FeatureNode[][] x;
        public Problem() {
                l = 0;
                n = 0;
                catmap = new CategoryMap<Integer>();
        }
        /**
         * Go loadira binarniot problem od fajl, pr. ima 2 klasi
         * @param filename imeto na fajlot koj go sodrzi problemot vo LibSVM format.
         */
        public void loadBinaryProblem(String filename) {
                String row;
                ArrayList<Integer> classes = new ArrayList<Integer>();
                ArrayList<FeatureNode []> examples = new ArrayList<FeatureNode []>();
                try {
                        BufferedReader r = new BufferedReader(new FileReader(filename));
                        while ((row = r.readLine()) != null) {
                                String [] elems = row.split(" ");
                                //Categorija:
                                Integer cat = Integer.parseInt(elems[0]);
                                catmap.addCategory(cat);
                                if (catmap.size() > 2) {
                                        throw new IllegalArgumentException("only 2 classes allowed!");
                                }
                                classes.add(catmap.getNewCategoryOf(cat));
                                //Index/value parovi:
                                examples.add(parseRow(elems));
                        }
                        x = new FeatureNode[examples.size()][];
                        y = new int[examples.size()];
                        for (int i=0; i<examples.size(); i++) {
                                x[i] = examples.get(i);
                                y[i] = 2*classes.get(i)-1; //0,1 => -1,1
                        }
                        l = examples.size();
                } catch (Exception e) {
                        System.out.println(e);
                }
        }
        /**
         * Parsira red od LibSVM format fajl.
         */
        public FeatureNode [] parseRow(String [] row) {
                FeatureNode [] example = new FeatureNode[row.length-1];
                int maxindex = 0;
                for (int i=1; i<row.length; i++) {
                        String [] iv = row[i].split(":");
                        int index = Integer.parseInt(iv[0]);
                        if (index <= maxindex) {
                                throw new IllegalArgumentException("indices must be in increasing order!");
                        }
                        maxindex = index;
                        double value = Double.parseDouble(iv[1]);
                        example[i-1] = new FeatureNode(index, value);
                }
                if (n < maxindex)
                        n = maxindex;
                return example;
        }
}