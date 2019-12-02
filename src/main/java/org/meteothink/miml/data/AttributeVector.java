package org.meteothink.miml.data;

import smile.math.MathEx;

/**
 * A vector with attribute information.
 *
 * @author Haifeng Li
 */
public class AttributeVector {

    /**
     * The attribute.
     */
    private Attribute attribute;

    /**
     * The data vector.
     */
    private double[] vector;

    /**
     * The optional names.
     */
    private String[] names;

    /**
     * Constructor.
     * @param attribute the attribute information.
     * @param vector the data vector.
     */
    public AttributeVector(Attribute attribute, double[] vector) {
        this.attribute = attribute;
        this.vector = vector;
    }

    /**
     * Constructor.
     * @param attribute the attribute information.
     * @param vector the data vector.
     * @param names optional names for each element.
     */
    public AttributeVector(Attribute attribute, double[] vector, String[] names) {
        this.attribute = attribute;
        this.vector = vector;
        this.names = names;
    }

    /**
     * Returns the attribute.
     */
    public Attribute attribute() {
        return attribute;
    }

    /**
     * Returns the data vector.
     */
    public double[] vector() {
        return vector;
    }

    /**
     * Returns the name vector.
     */
    public String[] names() {
        return names;
    }

    /**
     * Returns the vector size.
     * @return
     */
    public int size() {
        return vector.length;
    }
    @Override
    public String toString() {
        int n = 10;
        String s = head(n);
        if (vector.length <= n) return s;
        else return s + "\n" + (vector.length - n) + " more values...";
    }

    /** Shows the first few rows. */
    public String head(int n) {
        return toString(0, n);
    }

    /** Shows the last few rows. */
    public String tail(int n) {
        return toString(vector.length - n, vector.length);
    }

    /**
     * Stringify the vector.
     * @param from starting row (inclusive)
     * @param to ending row (exclusive)
     */
    public String toString(int from, int to) {
        StringBuilder sb = new StringBuilder();

        sb.append('\t');

        sb.append(attribute.getName());

        int end = Math.min(vector.length, to);
        for (int i = from; i < end; i++) {
            sb.append(System.getProperty("line.separator"));

            if (names != null) {
                sb.append(names[i]);
            } else {
                sb.append('[');
                sb.append(i + 1);
                sb.append(']');
            }
            sb.append('\t');

            if (attribute.getType() == Attribute.Type.NUMERIC)
                sb.append(String.format("%1.4f", vector[i]));
            else
                sb.append(attribute.toString(vector[i]));
        }

        return sb.toString();
    }

    /** Returns statistic summary. */
    public AttributeVector summary() {
        Attribute attr = new NumericAttribute(attribute.getName() + " Summary");
        String[] names = {"min", "q1", "median", "mean", "q3", "max"};

        double[] stat = new double[6];
        stat[0] = MathEx.min(vector);
        stat[1] = MathEx.q1(vector);
        stat[2] = MathEx.median(vector);
        stat[3] = MathEx.mean(vector);
        stat[4] = MathEx.q3(vector);
        stat[5] = MathEx.max(vector);

        return new AttributeVector(attr, stat, names);
    }
}