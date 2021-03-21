//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU

float preferredDistance = 0.25f; // d*
float repulsiveForce = 5;
float forceLimitConnectedPositive = 0.5;
float forceLimitConnectedNegative = 0.5;
float forceLimitDisconnectedNegative = -3;

bool spaceKeyPressed = false;
bool isHyperbolic = false;

const int numberOfNodes = 50;
const int visibleEdgesInPercent = 5;
long timeAtLastFrame = 0;

// for debugging purposes
void printVec3(std::string message, vec3 v) {
	printf("%s %9.6f, %9.6f, %9.6f\n", message.c_str(), v.x, v.y, v.z);
}

// preferredDistance < distance || connected
float magicFormula(float x) {
	return 3 * (((x - preferredDistance) * (x - preferredDistance)) / 2.6) * cosf((x - preferredDistance) / 2.6);
}

// distance < preferredDistance || connected
float magicFormula2(float x) {
	return -magicFormula(x) * repulsiveForce * repulsiveForce;
}

// not connected
float magicFormula3(float x) {
	return (-1 / (x * repulsiveForce));
}

// calculated the friction based on the elapsed time
float calculateFriction() {
	long timeFromStart = glutGet(GLUT_ELAPSED_TIME);
	return min(timeFromStart * timeFromStart / 100000000, 1);
}

// calculates the lorentz force between two vectors
float lorentzForce(vec3 p1, vec3 p2) {
	return p1.x * p2.x + p1.y * p2.y - p1.z * p2.z;
}

float getNormalizedDistanceSqr(vec3 from, vec3 to) {
		vec3 diff = from - to;
		return dot(diff, diff);
}

float getNormalizedDistance(vec3 from, vec3 to) {
	return sqrtf(getNormalizedDistanceSqr(from, to));
}

// calculates distance in the hyperbolic space
// swapped parameters according to the video
float getHyperbolicDistance(vec3 from, vec3 to) {
	return acoshf(-lorentzForce(to, from));
}

float getDistance(vec3 from, vec3 to) {
	/*if (!isHyperbolic)
		printf("You are not in the hyperbolic space!");
	return getHyperbolicDistance(from, to);*/
	return getNormalizedDistance(from, to);
}

//Hyperbolic coordinates to normalized device coordinates
	// x/z, y/z
vec3 hyperbolicToNDC(vec3 hyperbolic) {
	return hyperbolic / hyperbolic.z;
}

//normalized device coordinates to hyperbolic ones
	// 1 / (1-z^2)
vec3 NDCToHyperbolic(vec3 normalized) {
	return vec3(normalized.x, normalized.y, 1.0f) / sqrt(1.0f - (normalized.x) * (normalized.x) - (normalized.y) * (normalized.y));
}

// mirrors the node to the given point (source: hw video)
// m = p * cosh(d) + v * sinh(d)
vec3 mirror(vec3 p, vec3 m) {
	float d = getDistance(p, m);
	vec3 v = (m - p * coshf(d)) / sinhf(d);
	return p * coshf(2.0f * d) + v * sinhf(2.0f * d);
}

// calculates the midpoint of two points (source: hw video)
vec3 midPoint(vec3 p, vec3 q) {
	float d = getDistance(p, q);
	vec3 v = (q - p * coshf(d)) / sinhf(d);
	return p * coshf(d / 2.0f) + v * sinhf(d / 2.0f);
}

class Node {
	vec3 position;
	float mass;
	vec3 speed;
	vec3 acceleration;
	float pointSize;
	unsigned int nodeVBO; //vbo for the nodes

public:
	Node() {
		randomizePosition();
		mass = 1;
		speed = vec3(0, 0, 0);
		acceleration = vec3(0, 0, 0);
		pointSize = 10.0f;
	}

	Node(vec3 pos) : Node() {
		position = pos;
	}

	void randomizePosition() {
		// *2.0f is for the correct window size, the -1.0f is for the correct placement.
		//(without them the graph would be in the top-right corner)
		float x = -1;
		float y = -1;
		while (x * x + y * y >= 1) {
			x = (((float)rand() / RAND_MAX) * 2.0f) - 1.0f;
			y = (((float)rand() / RAND_MAX) * 2.0f) - 1.0f;
		}
		float z = 1;
		this->position = vec3(x, y, z);
	}

	vec3 getPosition() {
		return position;
	}

	void setPosition(vec3 position) {
		this->position = position;
	}

	vec3 getSpeed() {
		return speed;
	}

	void setMass(float mass) {
		this->mass = mass;
	}

	// moves the node in the hyperbolic space with the given vector
	void doubleMirror(vec3 m1, vec3 m2) {
		this->position = mirror(this->position, m1);
		this->position = mirror(this->position, m2);
	}

	void move(vec3 v) {
		this->position = this->position + v;
	}

	// scales the parameters of the node by lambda
	void scaleNode(float lambda) {
		this->position = this->position / lambda;
		this->speed = this->speed / lambda;
		this->acceleration = this->acceleration / lambda;
		this->pointSize = this->pointSize / lambda;
	}

	// applies the force on the node
	void applyForce(vec3 force, float deltaTime) { // kb 0.000040 nagyságrenű számok
		vec3 deltaSpeed = (force / mass) * deltaTime;
		this->speed = this->speed + deltaSpeed;
		this->speed = this->speed * (1 - calculateFriction());
	}

	// returns the magnitude of the force that we want to apply on two connected nodes
	float getForceMagnitudeConnected(vec3 other) {
		float distance = getDistance(this->position, other);
		if (distance > preferredDistance) {
			float magnitude = magicFormula(distance - preferredDistance);
			return min(magnitude, forceLimitConnectedPositive);
		}
		else {
			float magnitude = magicFormula2(distance);
			return max(magnitude, -forceLimitConnectedNegative);
		}
	}

	// returns the magnitude of the force that we want to apply on two disconnected nodes
	float getForceMagnitudeDisconnected(vec3 other) {
		float distance = getDistance(this->position, other);
		float magnitude = magicFormula3(distance);
		return max(magnitude, -forceLimitDisconnectedNegative);
	}

	// draws a node on the screen
	void draw() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &nodeVBO);
		glBindBuffer(GL_ARRAY_BUFFER, nodeVBO);

		std::vector<vec2> vertices;

		int sides = 20;
		float radius = 0.05f;

		//vec3 pos = normalizedToHyperbolic(this->position);

		/*for (int i = 0; i < sides; i++) {
			float hyperX = ((cosf(360.0f / sides * i * M_PI / 180.0f) * radius) + position.x);
			float hyperY = ((sinf(360.0f / sides * i * M_PI / 180.0f) * radius) + position.y);

			vertices.push_back(vec2(hyperX / position.z, hyperY / position.z));
		}*/

		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec2), vertices.data(), GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 1, 0, 1);
		float MVPtransf[4][4] = { 1, 0, 0, 0,
								  0, 1, 0, 0,
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);

		glPointSize(pointSize);
		glBindVertexArray(vao);
		//glDrawArrays(GL_POINTS, 0, 1);
		glDrawArrays(GL_TRIANGLE_FAN, 0, sides);
	}
};

class Edge {
	int node1;
	int node2;
	bool isVisible = false;
	unsigned int edgeVBO; //vbo for the edges

public:
	Edge(int nodeIndex1, int nodeIndex2, bool shouldDraw) {
		node1 = nodeIndex1;
		node2 = nodeIndex2;
		isVisible = shouldDraw;
	}

	void setVisible() {
		isVisible = true;
	}

	int getNode1() {
		return node1;
	}

	int getNode2() {
		return node2;
	}

	boolean getVisible() {
		return isVisible;
	}

	// drawn an edge on the screen
	void draw(vec3 p1, vec3 p2) {
		if (isVisible) {
			glGenVertexArrays(1, &vao);
			glBindVertexArray(vao);

			glGenBuffers(1, &edgeVBO);
			glBindBuffer(GL_ARRAY_BUFFER, edgeVBO);

			std::vector<vec2> vertices;

			vertices.push_back(vec2(p1.x, p1.y));
			vertices.push_back(vec2(p2.x, p2.y));

			glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec2), vertices.data(), GL_STATIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, 1, 1, 0);
			float MVPtransf[4][4] = { 1, 0, 0, 0,
									  0, 1, 0, 0,
									  0, 0, 1, 0,
									  0, 0, 0, 1 };

			location = glGetUniformLocation(gpuProgram.getId(), "MVP");
			glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, vertices.size());
		}
	}
};

class Graph {
private:
	std::vector<Node> nodes;
	std::vector<Edge> edges;
public:
	void addNode(Node n) {
		nodes.push_back(n);
	}

	void addEdge(Edge e) {
		edges.push_back(e);
	}

	// returns the number of edges that we want to draw on the screen
	int requiredEdgeCount() {
		return (numberOfNodes * (numberOfNodes - 1) / 2) * visibleEdgesInPercent / 100;
	}

	std::vector<Node> getNodes() {
		return nodes;
	}

	std::vector<Edge> getEdges() {
		return edges;
	}

	// calculates the degree of the given node
	int calculateDegree(int nodeIndex) {
		int degree = 0;
		for (int i = 0; i < edges.size(); i++) {
			if (edges[i].getNode1() == nodeIndex || edges[i].getNode2() == nodeIndex && edges[i].getVisible()) {
				degree++;
			}
		}
		return degree;
	}

	// calculates the hub of all nodes
	vec3 calculateHub() {
		vec3 hub = vec3(0, 0, 0);
		for each (Node node in nodes) {
			hub = hub + node.getPosition();
		}
		return hub / nodes.size();
	}

	// finds the farthest node on the screen
	// this helps us keep the graph on the screen
	float findExtremeCoordinates() {
		float m = 0;
		for (int i = 0; i < nodes.size(); i++)
		{
			if (abs(nodes[i].getPosition().x) > m)
				m = abs(nodes[i].getPosition().x);
			if (abs(nodes[i].getPosition().y) > m)
				m = abs(nodes[i].getPosition().y);
		}
		return m;
	}

	// sets required % of the nodes visible
	void chooseVisibleEdges() {
		int requiredNumberOfVisibleEdges = requiredEdgeCount();
		int visibleEdgesCount = 0;
		while (visibleEdgesCount < requiredNumberOfVisibleEdges) {
			int randomIndex = rand() % edges.size();
			if (edges[randomIndex].getVisible())
				continue;
			edges[randomIndex].setVisible();
			visibleEdgesCount++;
		}
	}

	void convertToHyperbolic() {
		isHyperbolic = true;
		for (int i = 0; i < nodes.size(); i++)
		{
			vec3 currentPosition = nodes[i].getPosition();
			vec3 newPosition = NDCToHyperbolic(currentPosition);
			nodes[i].setPosition(newPosition);
		}
	}

	void convertToNDC() {
		isHyperbolic = false;
		for (int i = 0; i < nodes.size(); i++)
		{
			vec3 currentPosition = nodes[i].getPosition();
			vec3 newPosition = hyperbolicToNDC(currentPosition);
			nodes[i].setPosition(newPosition);
		}
	}

	// creates a graph with the required number of visible edges
	void createGraph() {
		for (size_t i = 0; i < nodes.size() - 1; i++)
		{
			for (size_t j = i + 1; j < nodes.size(); j++)
			{
				if (i == j)
					continue;
				addEdge(Edge(i, j, false));
			}
		}
		chooseVisibleEdges();

		/*for (int i = 0; i < nodes.size(); i++)
			nodes[i].setMass((float)calculateDegree(i));*/
	}

	// moves the graph towards the center by using the hub function
	void moveTowardsCenter() {
		vec3 hub = -calculateHub();
		for (int i = 0; i < nodes.size(); i++) {
			nodes[i].move(hub);
		}
	}

	// draws the graph
	void draw() {
		for (int i = 0; i < edges.size(); i++)
		{
			edges[i].draw(nodes[edges[i].getNode1()].getPosition(), nodes[edges[i].getNode2()].getPosition());
		}
		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i].draw();
		}
	}

	// returns whether two nodes are connected or not
	boolean isConnected(int n1, int n2) {
		for (int i = 0; i < edges.size(); i++)
			if (edges[i].getNode1() == n1 && edges[i].getNode2() == n2 && edges[i].getVisible())
				return true;
		return false;
	}

	// calculates the force between two nodes
	vec3 calculateForceFrom(int current, int other) {
		vec3 diff = nodes[other].getPosition() - nodes[current].getPosition();
		if (this->isConnected(current, other))
			return normalize(diff) * nodes[current].getForceMagnitudeConnected(nodes[other].getPosition());
		return normalize(diff) * nodes[current].getForceMagnitudeDisconnected(nodes[other].getPosition());
	}

	// keeps the whole graph on the screen
	void keepItOnTheScreen() {
		float lambda = findExtremeCoordinates();
		for (int i = 0; i < nodes.size(); i++)
			nodes[i].scaleNode(lambda);
	}

	// summarises every force that influences the given node's position
	vec3 summariseForces(int nodeIndex) { //TODO
		vec3 sum = vec3(0, 0, 0);
		for (int i = 0; i < nodes.size(); i++)
		{
			if (nodeIndex == i)
				continue;
			sum = sum + calculateForceFrom(nodeIndex, i);
		}
		return sum;
	}

	// m = p * cosh(d) + v * sinh(d)
	void moveAllNodes(vec3 v) {
		vec3 m1 = vec3(0, 0, 1);
		vec3 m2 = vec3(0,0,1); // TODO

		for (int i = 0; i < nodes.size(); i++) {
			nodes[i].doubleMirror(m1, m2);
		}

	}

	// modifies the graph based on the passed time
	void modifyGraph(float deltaTime) {
		//convertToHyperbolic();
		for (int i = 0; i < nodes.size(); i++)
		{
			vec3 force = summariseForces(i);
			nodes[i].applyForce(force, deltaTime);
		}
		for (int i = 0; i < nodes.size(); i++) {
			vec3 v = nodes[i].getSpeed() * deltaTime;
			moveAllNodes(v);
		}
		//convertToNDC();

		moveTowardsCenter();
		keepItOnTheScreen();
	}
};

Graph graph;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	Node n1 = Node(vec3(-0.5f, 0.0f, 0));
	Node n2 = Node(vec3(0.5f, 0.0f, 0));
	Node n3 = Node(vec3(0.5f, 0.5f, 0));
	Edge e1 = Edge(0, 1, true);
	Edge e2 = Edge(0, 2, true);
	Edge e3 = Edge(2, 1, true);
	graph.addNode(n1);
	graph.addNode(n2);
	graph.addNode(n3);
	graph.addEdge(e1);


	for (int i = 0; i < numberOfNodes; i++) {
		graph.addNode(Node());
	}
	graph.createGraph();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	graph.draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	if (key == ' ') {
		timeAtLastFrame = glutGet(GLUT_ELAPSED_TIME);
		spaceKeyPressed = true;
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char* buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program

	float deltaTime = (float)(time - timeAtLastFrame) / 1000;

	if (spaceKeyPressed)
		graph.modifyGraph(min(deltaTime, 0.3));

	timeAtLastFrame = glutGet(GLUT_ELAPSED_TIME);

	glutPostRedisplay();
}
