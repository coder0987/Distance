
public class test {
    public static void main(String[] args) {
        EdgePath e = new EdgePath(new ByteTuple((byte)0,(byte)0), new ByteTuple((byte)10,(byte)10));
        for (byte i=0; i<20; i++) {
            e.edges.add(new ByteTuple(i,i));
        }
        e.wrap();
        for (ByteTuple b : e.edges) {
            System.out.println(b.first + ", " + b.second);
        }
        System.out.println("");
        e.wrap();
        for (ByteTuple b : e.edges) {
            System.out.println(b.first + ", " + b.second);
        }
    }
}
