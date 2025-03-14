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

    final static int IMG_SIZE = 784;
    final static int IMG_WIDTH = 28;

    public void clear() {
        System.out.println("CLEARING ALL IMAGE DATA");
        testImages = null;
        trainingImages = null;
        testReturn = null;
        trainingReturn = null;
        testEdgeReturn = null;
        trainingEdgeReturn = null;
    }

    public void debug(int i) {
        if (testLabels != null && testImages != null && testEdgeReturn != null) {
            debugPrint(testLabels[i], testImages[i], testEdgeReturn[i]);
            return;
        }
        if (testLabels != null) {
            System.out.println(testLabels[i]);
        }
        if (testImages != null) {
            printImg(testImages[i]);
        }
        if (testEdgeReturn != null) {
            printEdgePath(testEdgeReturn[i]);
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
        testLabels = new byte[pre_test_labels.length];
        System.arraycopy(pre_test_labels, 0, testLabels, 0, pre_test_labels.length);
        System.out.println("Loaded test labels");
    }

    public void loadTrainingLabels(byte[] pre_train_labels) {
        trainingLabels = new byte[pre_train_labels.length];
        System.arraycopy(pre_train_labels, 0, trainingLabels, 0, pre_train_labels.length);
        System.out.println("Loaded training labels");
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

    private static ByteTuple last(byte[][] image) {
        //Returns a potential first point for farthest geodesic distance
        for (byte row = IMG_WIDTH - 1; row >= 0; row--) {
            for (byte col = IMG_WIDTH - 1; col >= 0; col--) {
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

        while ((p = q.poll()) != null) {
            if (exploration[p.first][p.second] > farthest) {
                closest = p;
                farthest = exploration[p.first][p.second];
            }
            if (p.first > 0 && img[p.first - 1][p.second] != 0
                && exploration[p.first - 1][p.second] == 0) {
                exploration[p.first - 1][p.second] = (byte)(exploration[p.first][p.second] + 1);
                q.add(new ByteTuple((byte)(p.first - 1), p.second));
            }
            if (p.first < IMG_WIDTH - 1 && img[p.first + 1][p.second] != 0
                && exploration[p.first + 1][p.second] == 0) {
                exploration[p.first + 1][p.second] = (byte)(exploration[p.first][p.second] + 1);
                q.add(new ByteTuple((byte)(p.first + 1), p.second));
            }
            if (p.second > 0 && img[p.first][p.second - 1] != 0
                && exploration[p.first][p.second - 1] == 0) {
                exploration[p.first][p.second - 1] = (byte)(exploration[p.first][p.second] + 1);
                q.add(new ByteTuple(p.first, (byte)(p.second - 1)));
            }
            if (p.second < IMG_WIDTH - 1 && img[p.first][p.second + 1] != 0
                && exploration[p.first][p.second + 1] == 0) {
                exploration[p.first][p.second + 1] = (byte)(exploration[p.first][p.second] + 1);
                q.add(new ByteTuple(p.first, (byte)(p.second + 1)));
            }
        }
        return closest;
    }

    private static boolean connected(byte[][] img, ByteTuple start, ByteTuple end) {
        Queue<ByteTuple> q = new LinkedList<>();
        byte[][] exploration = new byte[IMG_SIZE][IMG_SIZE];
        
        exploration[start.first][start.second] = 1;

        q.add(start);

        ByteTuple p;

        while ((p = q.poll()) != null) {
            if (p.first == end.first && p.second == end.second) {
                return true;
            }
            if (p.first > 0 && img[p.first - 1][p.second] != 0
                && exploration[p.first - 1][p.second] == 0) {
                exploration[p.first - 1][p.second] = 1;
                q.add(new ByteTuple((byte)(p.first - 1), p.second));
            }
            if (p.first < IMG_WIDTH - 1 && img[p.first + 1][p.second] != 0
                && exploration[p.first + 1][p.second] == 0) {
                exploration[p.first + 1][p.second] = 1;
                q.add(new ByteTuple((byte)(p.first + 1), p.second));
            }
            if (p.second > 0 && img[p.first][p.second - 1] != 0
                && exploration[p.first][p.second - 1] == 0) {
                exploration[p.first][p.second - 1] = 1;
                q.add(new ByteTuple(p.first, (byte)(p.second - 1)));
            }
            if (p.second < IMG_WIDTH - 1 && img[p.first][p.second + 1] != 0
                && exploration[p.first][p.second + 1] == 0) {
                exploration[p.first][p.second + 1] = 1;
                q.add(new ByteTuple(p.first, (byte)(p.second + 1)));
            }
        }
        return false;
    }

    private static byte[][] bloat(byte[][] img) {
        byte[][] newImg = new byte[IMG_WIDTH][IMG_WIDTH];
        for (int row=0; row<IMG_WIDTH; row++) {
            for (int col=0; col<IMG_WIDTH; col++) {
                if (img[row][col] == 1 || ImageTools.activeBorder(col, row, img)) {
                    newImg[row][col] = 1;
                }
            }
        }
        return newImg;
    }

    /*
        Example edgeTransform
        00000
        11110
        11010
        00000
        
        EdgePath:
        
         ______
        |   _  |
        |__| |_|

        Represented as a 56x56 coordinate system
        0
        |1

        This edge is (3,1) <odd first number indicates vertical edge, as does odd second number

        Row, col rule: all values are stored (r,c) so y is first, not x



     */

    private static EdgePath edgeTransform(byte[][] img) {
        ByteTuple firstPoint = first(img);
        
        //Check for image integrity (that is, are all of the pixels connected?)
        ByteTuple lastPoint = last(img);
        while (!connected(img, firstPoint, lastPoint)) {
            img = bloat(img);
            firstPoint = first(img);
            lastPoint = last(img);
        }



        ByteTuple start = farthestBFS(img, firstPoint);
        ByteTuple end = farthestBFS(img, start);

        

        /*
        //Convert to edges
        start = pointToEdge(img, start);
        end = pointToEdge(img, end);
        */

        //Go all the way around
        EdgePath path = new EdgePath(start, end);

        /**
         * Start at start
         * Take the path down/left by default
         * Loop around until start is reached again
         * Watch for angles like such:
         * 1 0
         * 0 1
         */

        byte[][] exploredEdges = new byte[IMG_WIDTH][IMG_WIDTH];

        //DFS where allowed nodes must be active pixels bordered by inactive pixels (or on the edge of the image)
        path.produceEdgePathDFSDirected(img, exploredEdges, start, 0);

        if (!path.listIncludes(end, 1)) {
            System.out.println("End not present, retrying!");
            //End is not present, bloat the image and try again
            return edgeTransform(bloat(img));
        }

        //CAN USE produceEdgePathDFS OR produceEdgePathDFSDirected HERE

        return path;
    }

    @SuppressWarnings("CallToPrintStackTrace")
    public byte knnEdgeTransformed(byte[] args) {
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
        testEdgeReturn = new EdgePath[testImages.length];

        Thread[] threads = new Thread[testImages.length];
        for (int i=0; i<testImages.length; i++) {
            final int idx = i;

            threads[idx] = new Thread(() -> {
                testEdgeReturn[idx] = edgeTransform(testImages[idx]);
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

    @SuppressWarnings("CallToPrintStackTrace")
    public void edge_transform_training_images() {
        // Takes the images in testImages and outputs edgeTransformedTestImages
        if (trainingImages == null) {
            return;
        }
        if (trainingEdgeReturn != null) {
            //Cached results
            return;
        }
        trainingEdgeReturn = new EdgePath[trainingImages.length];

        Thread[] threads = new Thread[trainingImages.length];
        for (int i=0; i<trainingImages.length; i++) {
            final int idx = i;

            threads[idx] = new Thread(() -> {
                trainingEdgeReturn[idx] = edgeTransform(trainingImages[idx]);
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

        System.out.println("Edge transformed training images");
    }

    public void knn_evaluate(int K) {
        //Called by Python
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
                if (pixel == 0) {
                    System.out.print(" ");
                } else {
                    System.out.print((int) pixel);
                }
            }
            System.out.println();
        }
    }

    private static void printEdgePath(EdgePath e) {
        byte[][] img = new byte[IMG_WIDTH][IMG_WIDTH];
        for (ByteTuple b : e.edges) {
            img[b.first][b.second] = 1;
        }
        printImg(img);
    }

    private static void debugPrint(byte label, byte[][] img, EdgePath e) {
        byte[][] timg = new byte[IMG_WIDTH][IMG_WIDTH];
        for (ByteTuple b : e.edges) {
            timg[b.first][b.second] = 1;
        }

        System.out.println(label);
        for (int row=0; row<IMG_WIDTH; row++) {
            for (byte pixel : img[row]) {
                if (pixel == 0) {
                    System.out.print(" ");
                } else {
                    System.out.print((int) pixel);
                }
            }
            System.out.print(" | ");
            for (byte pixel : timg[row]) {
                if (pixel == 0) {
                    System.out.print(" ");
                } else {
                    System.out.print((int) pixel);
                }
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

    public boolean equals(ByteTuple o) {
        return first == o.first && second == o.second;
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

    //Check the pixels next to the current pixel first, then pixels at diagonal
    final static int[] xOrder = {0,1,0,-1, -1,1,1,-1, 2,2,2,2,2,   1,0,-1,   -2,-2,-2,-2,-2, -1,0,1};
    final static int[] yOrder = {1,0,-1,0, 1,-1,1,-1, 2,1,0,-1,-2, -2,-2,-2, -1, 0, 1, 2, 2, 2, 2, 2};

    //If going DOWN
    final static int[] dXOrder = {0,1,-1, 1,-1, 0,1,-1};
    final static int[] dYOrder = {1,1,1,  0,0,  -1,-1,-1};

    //If going UP
    final static int[] uXOrder = {0,1,-1,   1,-1, 0,1,-1};
    final static int[] uYOrder = {-1,-1,-1, 0,0,  1,1,1};

    //If going LEFT
    final static int[] lXOrder = {1,1,1,  0,0,  -1,-1,-1};
    final static int[] lYOrder = {1,0,-1, 1,-1, 1,0,-1};

    //If going RIGHT
    final static int[] rXOrder = {-1,-1,-1, 0,0,  1,1,1};
    final static int[] rYOrder = {1,0,-1,   1,-1, 1,0,-1};

    //May skip some extraneous pixels
    public void produceEdgePathDFS(byte[][] img, byte[][] explored, ByteTuple point) {
        explored[point.first][point.second] = 1;
        int x,y;
        for (int i=0; i<xOrder.length; i++) {
            x = xOrder[i];
            y = yOrder[i];
            if (ImageTools.inBounds(point, x, y) &&
                    img[point.first + y][point.second + x] == 1 &&
                    explored[point.first + y][point.second + x] == 0 && 
                    ImageTools.borderOrBlank(point.second + x, point.first + y, img)) {
                edges.add(point);
                point = new ByteTuple((byte) (point.first + y), (byte)(point.second + x));
                produceEdgePathDFS(img, explored, point);
                return;//Only want one path. May miss some pixels
            }
        }
        edges.add(point);//Add the last point on the path

        //Returns when no more edges can be added
    }

    //May skip some extraneous pixels
    public void produceEdgePathDFSDirected(byte[][] img, byte[][] explored, ByteTuple point, int direction) {
        explored[point.first][point.second] = 1;
        int x,y;

        int[] xs = switch(direction) {
            case 0: yield uXOrder;
            case 1: yield rXOrder;
            case 2: yield dXOrder;
            default: yield lXOrder;
        };

        int[] ys = switch(direction) {
            case 0: yield uYOrder;
            case 1: yield rYOrder;
            case 2: yield dYOrder;
            default: yield lYOrder;
        };

        for (int i=0; i<xs.length; i++) {
            x = xs[i];
            y = ys[i];
            if (ImageTools.inBounds(point, x, y) &&
                    img[point.first + y][point.second + x] == 1 &&
                    explored[point.first + y][point.second + x] == 0 && 
                    ImageTools.borderOrBlank(point.second + x, point.first + y, img)) {
                edges.add(point);
                point = new ByteTuple((byte) (point.first + y), (byte)(point.second + x));
                produceEdgePathDFSDirected(img, explored, point, (y==-1 ? 0 : x==1 ? 1 : y==1 ? 2 : 3));
                return;//Only want one path. May miss some pixels
            }
        }

        //Reach far incase we got stuck
        for (int i=8; i<xOrder.length; i++) {
            x = xOrder[i];
            y = yOrder[i];
            if (ImageTools.inBounds(point, x, y) &&
                    img[point.first + y][point.second + x] == 1 &&
                    explored[point.first + y][point.second + x] == 0 && 
                    ImageTools.borderOrBlank(point.second + x, point.first + y, img)) {
                edges.add(point);
                point = new ByteTuple((byte) (point.first + y), (byte)(point.second + x));
                produceEdgePathDFSDirected(img, explored, point, (y==-1 ? 0 : x==1 ? 1 : y==1 ? 2 : 3));
                return;
            }
        }


        edges.add(point);//Add the last point on the path

        //Returns when no more edges can be added
    }

    public boolean listIncludes(ByteTuple point, int distance) {
        for (ByteTuple b : edges) {
            if (point.distance(b.first, b.second) <= distance) {return true;}
        } 
        return false;
    }
}

class ImageTools {
    public static boolean inBounds(ByteTuple point, int x, int y) {
        //Assuming bounds to be 0, IMG_WIDTH
        int bx = point.second + x;
        int by = point.first + y;
        return bx >= 0 && by >= 0 && bx < Distance.IMG_WIDTH && by < Distance.IMG_WIDTH;
    }
    public static boolean borderOrBlank(int x, int y, byte[][] img) {
        //Returns true if (x,y) is on the edge or bordered by an inactive pixel
        return x==0 || y==0 || x==Distance.IMG_WIDTH-1 || y==Distance.IMG_WIDTH-1 ||
            img[y][x-1] == 0 || img[y][x+1] == 0 || img[y-1][x] == 0 || img[y+1][x] == 0;
    }
    public static boolean activeBorder(int x, int y, byte[][] img) {
        //Returns true if (x,y) is  bordered by an active pixel
        return  (x > 0 && img[y][x-1] == 1) || (x < Distance.IMG_WIDTH - 1 && img[y][x+1] == 1) || (y > 0 && img[y-1][x] == 1) || (y < Distance.IMG_WIDTH  - 1 && img[y+1][x] == 1);
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