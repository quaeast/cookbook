# Java

[oracle java doc](https://docs.oracle.com/javase/specs/index.html)

## 面向对象

### 方法的 overload 和 override

方法签名（method signature）：一般指方法的名字和参数列表

`overload`方法名相同，但是参数不同，返回值也可以不同
`override`通常在继承时发生，方法名相同，参数列表相同，返回值也必须相同

## 函数式编程	

### 函数式接口 FunctionalInterface

```java
//1. 函数式接口：只包含一个抽象方法，不包括default和static方法
@FunctionalInterface
interface ILike {
    void lambda();
}
```

`FunctionalInterface`允许传入：

* 接口的实现类（传统写法，代码较繁琐）；
* Lambda表达式（只需列出参数名，由编译器推断类型）；
* 符合方法签名的静态方法；
* 符合方法签名的实例方法（实例类型被看做第一个参数类型）；
* 符合方法签名的构造方法（实例类型被看做返回类型）。

`FunctionalInterface`**不强制继承关系**，不需要方法名称相同，只要求方法参数（类型和数量）与方法返回类型相同，即认为**方法签名相同**。

### lambda

```java
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

//1. 函数式接口：只包含一个抽象方法，不包括default和static方法
@FunctionalInterface
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

### 方法引用

```java
ClassName::methodName
ClassName::new //构造方法引用
```

## 注解

> 什么是注解（Annotation）？注解是放在Java源码的类、方法、字段、参数前的一种特殊“注释”

定义一个注解时，还可以定义配置参数。配置参数可以包括：

* 所有基本类型；
* String；
* 枚举类型；
* 基本类型、String、Class以及枚举的数组。

```java
@Inherited
@Repeatable(Reports.class)
@Retention(RetentionPolicy.RUNTIME)//默认为class，通常需要声明
@Target(ElementType.METHOD)
public @interface Report {
    int type() default 0;
    String level() default "info";
    String value() default "";
}
```


## 设计模式

设计模式包括：

* **创建型模式**：创建型模式关注点是如何创建对象，其核心思想是要把对象的创建和使用相分离，这样使得两者能相对独立地变换
* **结构型模式**：
* **行为型模式**：

#### 开闭原则（Open Closed Principle）

软件应该对扩展开放，而对修改关闭。

#### 里氏替换原则（Barbara Liskov）

任何基类可以出现的地方，子类一定可以出现。即如果我们调用一个父类的方法可以成功，那么替换成子类调用也应该完全可以运行。

这里其实也符合开闭原则，子类只对父类的延伸，拥有父类包括的一切能力。这一点在Java中需要注意，声明对应的类型并不会改变对象的行为。

```java
package cn.quaeast;

class Father{
    public String hello(){
        return "Fhello";
    }
}

class Son extends Father {
    public String hello(){
        return "Shello";
    }
}

public class TestPC {
    public static void main(String[] args) {
        //声明对应的类型并不会改变对象的行为
        Father f = new Father();
        System.out.println(s.hello());//Fhellp
        f = new Son();
        System.out.println(s.hello());//Shellp
    }
}
```

#### 工厂方法（Factory Method）：创建型模式

>  定义一个用于创建对象的接口，让子类决定实例化哪一个类。Factory Method使一个类的实例化**延迟到其子类**。

实际上大多数情况下我们并不需要抽象工厂，而是通过**静态方法**直接返回产品。这种简化的使用静态方法创建产品的方式称为**静态工厂方法（Static Factory Method）**。

```java
List<String> list = List.of("A", "B", "C"); //静态工厂方法
```

#### 抽象工厂（Abstract Factory）：创建型模式

> 提供一个创建一系列相关或相互依赖对象的接口，而无需指定它们具体的类。

定义抽象的接口：

```java
public interface AbstractFactory {
    // 创建Html文档:
    HtmlDocument createHtml(String md);
    // 创建Word文档:
    WordDocument createWord(String md);
}

// Html文档接口:
public interface HtmlDocument {
    String toHtml();
    void save(Path path) throws IOException;
}

// Word文档接口:
public interface WordDocument {
    void save(Path path) throws IOException;
}
```

实现抽象接口：

```java
public interface AbstractFactory {
    public static AbstractFactory createFactory(String name) {
        if (name.equalsIgnoreCase("fast")) {
            return new FastFactory();
        } else if (name.equalsIgnoreCase("good")) {
            return new GoodFactory();
        } else {
            throw new IllegalArgumentException("Invalid factory name");
        }
    }
}
```

#### 生成器（Builder）：创建型模式

> 将一个复杂对象的构建与它的表示分离（我认为应该翻译成把复杂对象构建的表示分离），使得同样的构建过程可以创建不同的表示。

```java
String url = URLBuilder.builder() // 创建Builder
        .setDomain("www.liaoxuefeng.com") // 设置domain
        .setScheme("https") // 设置scheme
        .setPath("/") // 设置路径
        .setQuery(Map.of("a", "123", "q", "K&R")) // 设置query
        .build(); // 完成build
        
```

#### 原型（Prototype）：创建型模式

> 用原型实例指定创建对象的种类，并且通过拷贝这些原型创建新的对象。

在 JavaScript 中常见

#### 单例（Singleton）：创建型模式

> 保证一个类仅有一个实例，并提供一个访问它的全局访问点。

在Java中一般使用约定的形式而不主动约束

#### 结构型模式


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

#### synchronized

`synchronized`关键词可以作为方法的关键字用于锁住该类的`this`，如果是**静态**方法则锁住该类`class`。

```java
class MyClass {
    synchronized void myMethod(){}
    
    synchronized static void myStaticMethod(){}
    
    void myMethod2(){
        synchronized(this){}
    }
}
```

#### lock

```java
public class TestLock {
    private final ReentrantLock lock = new ReentrantLock();

    public void add(int n) {
        lock.lock();
        try {
            for (int i = 0; i < n; i++) {
                balance = balance + 1;
            }
        } finally {
            lock.unlock();
        }
    }
}
```

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

生产者和消费者问题是指生产者和消费者共享用一个资源，并且生产者和消费者之间相互依赖互为条件。

* 对于生产者，没有生产产品之前，要通知消费者等待。而生产了产品之后，又需要马上通知消费者消费
* 对于消费者，在消费之后，要通知生产者已经结束消费，需要生产新的产品以供消费
* 在生产者消费者问题中，**仅有synchronized是不够的**
    * synchronized 可阻止并发更新同一个共享资源，实现了同步
    * synchronized 不能用来实现不同线程之间的消息传递 （通信）

```java
//一下方法只能在同步方法或同步代码块中使用，否则会抛出IllegalMonitorStateException
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