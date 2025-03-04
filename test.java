public class test {
    public static void main(String[] args) {
        byte one = (byte) 0x0f;
        byte two = (byte) 0xff;
        System.out.println(Byte.compareUnsigned(one, two) > 0);
        byte smaller = (Byte.compareUnsigned(one, two) > 0 ? two : one);
        System.out.println(smaller);
    }
}
