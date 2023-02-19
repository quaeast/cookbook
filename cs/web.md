# Java

## lambda

```
package cn.quaeast;

//函数式接口的各种实现方法

public class TestLambda {
    //3.静态内部类
    static class Like2 implements ILike {
        @Override
        public void lambda() {
            System.out.println("I like lambda2");
        }
    }


    public static void main(String[] args) {
        ILike like = new Like();
        like.lambda();

        like = new Like2();
        like.lambda();

        //4.局部内部类
        class Like3 implements ILike {
            @Override
            public void lambda() {
                System.out.println("I like lambda3");
            }
        }
        like = new Like3();
        like.lambda();

        //5.匿名内部类
        like = new ILike() {
            @Override
            public void lambda() {
                System.out.println("I like lambda4");
            }
        };
        like.lambda();

        //6.用lambda简化
        like = () -> System.out.println("I like lambda");
        like.lambda();
    }
}

//1. 函数式接口：一个方法只包括一个接口
interface ILike {
    void lambda();
}

//2.实现类
class Like implements ILike {
    @Override
    public void lambda() {
        System.out.println("I like lambda");
    }
}
```

## Java 多线程基础

### 线程的创建

* Thread 类
* Runnable 接口
* Callable 接口

继承`Thread`

```java
package cn.quaeast;

public class TestThread extends Thread {
    @Override
    public void run() {
        for (int i = 0; i < 2000; i++) {
            System.out.println(Thread.currentThread().getName() + " : " + i);
        }
    }

    public static void main(String[] args) {
        TestThread testThread = new TestThread();
        testThread.start();
    }
}
```

```java
package cn.quaeast;

//实现 `Runnable` 接口。因为Java只支持单继承的局限性，所以实现`Runnable`更实用

public class TestThread3 implements Runnable {
    @Override
    public void run() {
        for (int i = 0; i < 200; i++) {
            System.out.println(Thread.currentThread().getName() + " : " + i);
        }
    }

    public static void main(String[] args) {
        TestThread3 testThread3 = new TestThread3();
        Thread thread = new Thread(testThread3);
        thread.start();
    }
}
```

```java
package cn.quaeast;

//多线程网图下载

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.net.URL;

public class TestThread2 extends Thread {
    private String url;
    private String name;

    public TestThread2(String url, String name) {
        this.url = url;
        this.name = name;
    }

    @Override
    public void run() {
        WebDownloader webDownloader = new WebDownloader();
        webDownloader.downloader(url, name);
        System.out.println("The downloaded file name is: " + name);
    }

    public static void main(String[] args) {
        TestThread2 t1 = new TestThread2("https://picx.zhimg.com/v2-d6921962f485cd70e5c66f20852c002f_b.jpg", "t1.jpg");
        TestThread2 t2 = new TestThread2("https://picx.zhimg.com/v2-d6921962f485cd70e5c66f20852c002f_b.jpg", "t2.jpg");
        TestThread2 t3 = new TestThread2("https://picx.zhimg.com/v2-d6921962f485cd70e5c66f20852c002f_b.jpg", "t3.jpg");

        t1.start();
        t2.start();
        t3.start();
    }

}

class WebDownloader{
    public void downloader(String url, String name){
        try {
            FileUtils.copyURLToFile(new URL(url), new File(name));
        } catch (IOException e){
            e.printStackTrace();
            System.out.println("IO error, downloader() method error");
        }
    }
}
```

Callable

```java
package cn.quaeast;

import java.util.concurrent.*;

//可以定义返回值
//可以抛出异常

public class TestCallable implements Callable<Boolean> {
    private String url;
    private String name;

    public TestCallable(String url, String name) {
        this.url = url;
        this.name = name;
    }

    @Override
    public Boolean call() {
        WebDownloader webDownloader = new WebDownloader();
        webDownloader.downloader(url, name);
        System.out.println("download file name: "+name);
        return true;
    }

    public static void main(String[] args) throws ExecutionException, InterruptedException {
        TestCallable t1 = new TestCallable("https://picx.zhimg.com/v2-d6921962f485cd70e5c66f20852c002f_b.jpg", "t1.jpg");
        TestCallable t2 = new TestCallable("https://picx.zhimg.com/v2-d6921962f485cd70e5c66f20852c002f_b.jpg", "t2.jpg");
        TestCallable t3 = new TestCallable("https://picx.zhimg.com/v2-d6921962f485cd70e5c66f20852c002f_b.jpg", "t3.jpg");

        ExecutorService ser = Executors.newFixedThreadPool(3);
        Future<Boolean> result1 = ser.submit(t1);
        Future<Boolean> result2 = ser.submit(t2);
        Future<Boolean> result3 = ser.submit(t3);

        Boolean rs1 = result1.get();
        Boolean rs2 = result2.get();
        Boolean rs3 = result3.get();
    }
}
```

### 中断线程

目前Java线程只能从线程内主动中断而不能被动中断。使用的方法为`thread.interrupt()`，该方法只是设置interrupt位，不会主动中断线程。在阻塞时调用`thread.interrupt()`会抛出异常`InterruptException`

想要中断线程需要在线程内使用：

```java
Thread.currentThread().isInterrupted();
Thread.interrupted();//该静态方法调用后当前Thread实例的interrupted位会被置为False
```

### 线程的状态

线程有以下几种状态：

* New
* Runnable
* Blocked
* Waiting
* Timed waiting
* Terminated

```
//线程睡眠
Thread.sleep();
//礼让，当前进程暂停但是不阻塞，不一定成功
//A hint to the scheduler that the current thread is willing to yield its current use of a processor. The scheduler is free to ignore this hint.
Thread.yield();
//Waits for this thread to die.
thread.join();
```

### 线程属性

* 优先级
* 守护进程，在是剩下守护进程时虚拟机退出

```
thread.setPriority(int);
thread.setDaemon(boolean);
//需要在start之前设置
```

### 锁

#### 死锁

并发下，线程因为相互等待对方资源，导致“永久”阻塞的现象。

比如线程1锁住锤子需要钉子，线程2锁住钉子需要锤子，这就造成了死锁。

**造成死锁的四大因素：**

1. 互斥：一个资源每次只能被一个线程使用 -- 互斥锁
2. 占有且等待：一个线程在阻塞等待其他现成持有的资源，不释放已占有资源 -- 等待对方释放资源
3. 不可抢占：资源只能由持有他的资源资源释放，不能被强行剥夺 -- 无法释放对方资源
4. 循环等待：若干线程形成尤为相接的循环等待资源关系 -- 两个线程互相等待

想要破解死锁只需要破坏四大因素即可

**开发过程中可以使用如下方法解决：**

1. 破坏1：不使用互斥锁，使用原子操作或乐观锁
2. 破坏2：把多个资源合并然后一次锁住
3. 破坏3：注意加锁的顺序，保证每个线程按照同样的顺序加锁，如Lock
2. 要注意加锁时限，设置一个超时时间
3. 注意死锁检查，预防死锁发生

**如何定位死锁：**

```shell
//查看java进程
jps
//死锁检查命令
jstack <pid>
```

### 生产者和消费者: Object的Wait()

生产者和消费者问题中，仅有Synchronized是不够的

```
object.wait()
object.wait(long timeout)
object.notify()
object.notifyall()

```



# Mysql

## 事务

**事务的隔离级别 ACID**：

* 原子性：同时成功，同时失败。原子性由undo log实现
* **一致性**：使用事务的最终目的，一般由代码实现
* 隔离性：在事务并发执行时，他们内部操作互不干扰
* 持久性：一旦提交了事务，它对数据库的改变就应该是永久性的，持久性由redo log日志来包装

**事务隔离级别**：

* read uncommit: 脏读
* read commit: 不可重复读（oracle默认）
* repeatable read: 幻读（mysql默认）
* serializable: 上面的问题全部解决

隔离性由SQL的各种锁和MVCC机制实现。、