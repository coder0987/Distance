import py4j.GatewayServer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Queue;

public class Distance {
    byte[][][] testImages;
    byte[][][] trainingImages;

    byte[] testLabels;
    byte[] trainingLabels;

    byte[] testReturn;
    byte[] trainingReturn;

    EdgePath[] testEdgeReturn;
    EdgePath[] trainingEdgeReturn;

    byte[] dtwEdgeTest;

    final static int IMG_SIZE = 784;
    final static int IMG_WIDTH = 28;

    public void clear() {
        System.out.println("CLEARING ALL IMAGE DATA");
        testImages = null;
        trainingImages = null;
        testReturn = null;
        trainingReturn = null;
    }

    public void debug(int i) {
        if (testImages != null) {
            printImg(testImages[i]);
        }
    }

    public void loadTestData(byte[] pre_imgs) {
        int numImgs = pre_imgs.length / IMG_SIZE;
        System.out.println("Loading " + numImgs + " test images");
        testImages = new byte[numImgs][IMG_WIDTH][IMG_WIDTH];
        for (int i=0; i<numImgs; i++) {
            for (int row=0; row<IMG_WIDTH; row++) {
                for (int col=0; col<IMG_WIDTH; col++) {
                    testImages[i][row][col] = pre_imgs[i * IMG_SIZE + row * IMG_WIDTH + col];
                }
            }
        }
        System.out.println("Loaded test images");
    }
    public void loadTrainingData(byte[] pre_train_imgs) {
        int numImgs = pre_train_imgs.length / IMG_SIZE;
        System.out.println("Loading " + numImgs + " training images");
        trainingImages = new byte[numImgs][IMG_WIDTH][IMG_WIDTH];
        for (int i=0; i<numImgs; i++) {
            for (int row=0; row<IMG_WIDTH; row++) {
                for (int col=0; col<IMG_WIDTH; col++) {
                    trainingImages[i][row][col] = pre_train_imgs[i * IMG_SIZE + row * IMG_WIDTH + col];
                }
            }
        }
        System.out.println("Loaded training images");
    }

    public void loadTestLabels(byte[] pre_test_labels) {
        System.arraycopy(pre_test_labels, 0, testLabels, 0, pre_test_labels.length);
    }

    public void loadTrainingLabels(byte[] pre_train_labels) {
        System.arraycopy(pre_train_labels, 0, trainingLabels, 0, pre_train_labels.length);
    }

    @SuppressWarnings("CallToPrintStackTrace")
    public byte[] distance_transform() {
        if (testImages == null) {
            return null;
        }
        if (testReturn != null) {
            //Cached results
            return testReturn;
        }
        testReturn = new byte[testImages.length * IMG_SIZE];

        Thread[] threads = new Thread[testImages.length];
        for (int i=0; i<testImages.length; i++) {
            final int idx = i;

            threads[idx] = new Thread(() -> {
                System.arraycopy(Distance.transform(testImages[idx]), 0, testReturn, idx * IMG_SIZE, IMG_SIZE);
            });
            threads[i].start();
        }

        try {
            for (Thread thread : threads) {
                thread.join();
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println("Transformed test images");
        return testReturn;
    }
    @SuppressWarnings("CallToPrintStackTrace")
    public byte[] distance_transform_training() {
        if (trainingImages == null) {
            return null;
        }
        if (trainingReturn != null) {
            //Cached results
            return trainingReturn;
        }
        trainingReturn = new byte[trainingImages.length * IMG_SIZE];

        Thread[] threads = new Thread[trainingImages.length];
        for (int i=0; i<trainingImages.length; i++) {
            final int idx = i;

            threads[idx] = new Thread(() -> {
                System.arraycopy(Distance.transform(trainingImages[idx]), 0, trainingReturn, idx * IMG_SIZE, IMG_SIZE);
            });
            threads[i].start();
        }

        try {
            for (Thread thread : threads) {
                thread.join();
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        
        System.out.println("Transformed training images");
        return trainingReturn;
    }

    private static byte[][] transform(byte[][] image) {
        ArrayList<ByteTuple> active = new ArrayList<>();
        byte[][] transformedImage = new byte[28][28];
        
        //Parallelizable
        for (byte row = 0; row < 28; row++) {
            for (byte col = 0; col < 28; col++) {
                if (image[row][col] == 1) {
                    active.add(new ByteTuple(row, col));
                }
            }
        }

        byte d;
        byte dt;
        
        //Parallelizable
        for (byte row = 0; row < 28; row++) {
            for (byte col = 0; col < 28; col++) {
                if (image[row][col] == 1) {
                    //Skip active points
                    continue;
                }
                d = (byte) 0xff;

                for (ByteTuple point : active) {
                    dt = point.distance(row, col);
                    d = (Byte.compareUnsigned(d, dt) > 0 ? dt : d);
                }
                transformedImage[row][col] = d;
            }
        }
        return transformedImage;
    }

    private static ByteTuple first(byte[][] image) {
        //Returns a potential first point for farthest geodesic distance
        for (byte row = 0; row < IMG_WIDTH; row++) {
            for (byte col = 0; col < IMG_WIDTH; col++) {
                if (image[row][col] > 0) {
                    return new ByteTuple(row, col);
                }
            }
        }
        return null;
    }

    private static ByteTuple farthestBFS(byte[][] img, ByteTuple start) {
        Queue<ByteTuple> q = new LinkedList<>();
        byte[][] exploration = new byte[IMG_SIZE][IMG_SIZE];

        ByteTuple closest = start;
        
        exploration[start.first][start.second] = 1;
        byte farthest = 1;

        q.add(start);

        ByteTuple p;

        while ((p = q.remove()) != null) {
            if (exploration[p.first][p.second] > farthest) {
                closest = p;
                farthest = exploration[p.first][p.second];
            }
            if (p.first > 0 && img[p.first - 1][p.second] != 0
                && exploration[p.first - 1][p.second] == 0) {
                exploration[p.first - 1][p.second] = (byte)(exploration[p.first][p.second] + 1);
                q.add(new ByteTuple((byte)(p.first - 1), p.second));
            }
            if (p.first < IMG_WIDTH && img[p.first + 1][p.second] != 0
                && exploration[p.first + 1][p.second] == 0) {
                exploration[p.first + 1][p.second] = (byte)(exploration[p.first][p.second] + 1);
                q.add(new ByteTuple((byte)(p.first + 1), p.second));
            }
            if (p.second > 0 && img[p.first][p.second - 1] != 0
                && exploration[p.first][p.second - 1] == 0) {
                exploration[p.first][p.second - 1] = (byte)(exploration[p.first][p.second] + 1);
                q.add(new ByteTuple(p.first, (byte)(p.second - 1)));
            }
            if (p.second < IMG_WIDTH && img[p.first][p.second + 1] != 0
                && exploration[p.first][p.second + 1] == 0) {
                exploration[p.first][p.second + 1] = (byte)(exploration[p.first][p.second] + 1);
                q.add(new ByteTuple(p.first, (byte)(p.second + 1)));
            }
        }
        return closest;
    }

    private static EdgePath edgeTransform(byte[][] img) {
        ByteTuple firstPoint = first(img);
        ByteTuple start = farthestBFS(img, firstPoint);
        ByteTuple end = farthestBFS(img, start);

        //Go all the way around
        EdgePath path = new EdgePath(start, end);

        //TODO

        

        return path;
    }

    @SuppressWarnings("CallToPrintStackTrace")
    public byte knn_edge_transformed(byte[] args) {
        byte index = args[0];
        byte k = args[1];
        //Find knn for test against all training data

        //All are transformed and stored in EdgePath[] testEdgeReturn and trainingEdgeReturn

        EdgePath edge = testEdgeReturn[index];

        //Step 1: test against each training and store distance
        ByteTuple[] distNum = new ByteTuple[trainingEdgeReturn.length];
        Thread[] threads = new Thread[trainingEdgeReturn.length];

        for (int i=0; i<testImages.length; i++) {
            final int idx = i;
            distNum[idx].second = trainingLabels[idx];

            threads[idx] = new Thread(() -> {
                distNum[idx].first = Distance.dynamicTimeWarp(edge, trainingEdgeReturn[idx]);
            });
            threads[i].start();
        }

        try {
            for (Thread thread : threads) {
                thread.join();
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        //Step 2: find the largest k values
        PriorityQueue<ByteTuple> min = new PriorityQueue<>(new ByteTupleComparator());
        for (ByteTuple it : distNum) {
            min.add(it);
            if (min.size() > k) {
                min.remove();
            }
        }


        //Step 3: return the most common among the top k
        byte[] top = new byte[10];
        byte best = 0;
        for (ByteTuple it : min) {
            top[it.second]++;
            if (top[it.second] > top[best]) {
                best = it.second;
            }
        }

        return best;
    }

    @SuppressWarnings("CallToPrintStackTrace")
    public void edge_transform_test_images() {
        // Takes the images in testImages and outputs edgeTransformedTestImages
        if (testImages == null) {
            return;
        }
        if (testEdgeReturn != null) {
            //Cached results
            return;
        }
        testEdgeReturn = new EdgePath[testImages.length * IMG_SIZE];

        Thread[] threads = new Thread[testImages.length];
        for (int i=0; i<testImages.length; i++) {
            final int idx = i;

            threads[idx] = new Thread(() -> {
                //TODO
                System.arraycopy(Distance.edgeTransform(testImages[idx]), 0, testEdgeReturn, idx * IMG_SIZE, IMG_SIZE);
            });
            threads[i].start();
        }

        try {
            for (Thread thread : threads) {
                thread.join();
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println("Edge transformed test images");
    }

    private static byte dynamicTimeWarp(EdgePath e1, EdgePath e2) {
        //Use DTW to find the distance between the two edgepaths
        /**
         * Step 1: Determine endpoints
         * Step 2: DTW loop around the whole edge path
         * Step 3: Return the calculated distance as a byte
         */


        return 0;
    }

    public static void printImg(byte[][] img) {
        for (byte[] row : img) {
            for (byte pixel : row) {
                System.out.print((int) pixel);
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        Distance app = new Distance();
        // app is now the gateway.entry_point
        GatewayServer server = new GatewayServer(app);
        server.start();

        System.out.println("Running!");
    }
}

class ByteTuple {
    public byte first;
    public byte second;
    public ByteTuple(byte x, byte y) {
        first = x;
        second = y;
    }
    public ByteTuple() {
        first = 0;
        second = 0;
    }
    public byte distance(byte x, byte y) {
        return (byte) Math.sqrt(Math.pow(first - x, 2) + Math.pow(second - y, 2));
    }
}

class EdgePath {
    public ArrayList<ByteTuple> edges;
    public ByteTuple start;
    public ByteTuple end;
    public EdgePath(ByteTuple first, ByteTuple mid) {
        start = first;
        end = mid;
        edges = new ArrayList<>();
    }
}

class ByteTupleComparator implements Comparator<ByteTuple>
{
    @Override
    public int compare(ByteTuple i1, ByteTuple i2)
    {
        return Byte.compareUnsigned(i1.first, i2.first);
    }
}