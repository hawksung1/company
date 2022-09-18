# Version 20220521
> ctrl + f

# file, file read, file print
## Text File, text_file, textfile
``` java
```
## Binary File, binary_file, binaryfile

# cmd input, command line input, cli input, scanner
## input을 queue에 저장, 출력
```java
import java.util.ArrayList;
import java.util.Scanner;

public class RunManager {
	public static ArrayList<String> queue = new ArrayList<>();
	public static void main(String[] args) {
		try (Scanner fs = new Scanner(System.in)) {
			while(fs.hasNextLine()) {
				String userinputLine = fs.nextLine();
				
				String[] splitLine = userinputLine.split(" ");
				if("SEND".equals(splitLine[0])) {
					queue.add(splitLine[1]);
				} else if("RECEIVE".equals(splitLine[0])) {
					if(queue.size()>0) {
						System.out.println(queue.get(0));
						queue.remove(0);
					}
				}
			}
		}
	}
}

```
# json handle, stream to json, request to json
``` java
public Map<String, Object> getJsonFromReqeustBody(HttpServletRequest request) throws IOException {
        
    String body = null;
    StringBuilder stringBuilder = new StringBuilder();
    BufferedReader bufferedReader = null;

    try {
        InputStream inputStream = request.getInputStream();
        if (inputStream != null) {
            bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
            char[] charBuffer = new char[128];
            int bytesRead = -1;
            while ((bytesRead = bufferedReader.read(charBuffer)) > 0) {
                stringBuilder.append(charBuffer, 0, bytesRead);
            }
        }
    } catch (IOException ex) {
        throw ex;
    } finally {
        if (bufferedReader != null) {
            try {
                bufferedReader.close();
            } catch (IOException ex) {
                throw ex;
            }
        }
    }

    body = stringBuilder.toString();

    Gson gson = new Gson();
    Map<String,Object> map = new HashMap<String,Object>();
    map = (Map<String,Object>) gson.fromJson(bodyJson, map.getClass());

    return map;
}
```



# server
## http, servlet
``` java
import org.eclipse.jetty.server.*;
import org.eclipse.jetty.servlet.ServletHandler;

public class MyServer {

	public static void main(String[] args) throws Exception {
		new MyServer().start();
	}

	public void start() throws Exception {
		Server server = new Server();
		ServerConnector http = new ServerConnector(server);
		http.setHost("127.0.0.1");
		http.setPort(8080);
		server.addConnector(http);

		ServletHandler servletHandler = new ServletHandler();
		servletHandler.addServletWithMapping(MyServlet.class, "/mypath");
		server.setHandler(servletHandler);

		server.start();
		server.join();
	}
}
```
``` java
import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.http.*;

public class MyServlet extends HttpServlet {

	private static final long serialVersionUID = 1L;

	protected void doGet(HttpServletRequest req, HttpServletResponse res) throws ServletException, IOException {
		res.setStatus(200);
		res.getWriter().write("Hello!");
	}
}
```


