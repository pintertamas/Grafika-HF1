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

float preferredDistance = 0.25f;
//float attraction = 0.005;
float repulsiveForce = 5;
float forceLimitConnectedPositive = 0.5;
float forceLimitConnectedNegative = 0.5;
float forceLimitDisconnectedNegative = -3;

bool spaceKeyDown = false;
long spaceKeyTime = 0;

const int numberOfNodes = 50;
const int visibleEdgesInPercent = 5;
long timeAtLastFrame = 0;

// for debugging purposes
void printVec3(std::string message, vec3 v) {
	printf("%s %9.6f, %9.6f, %9.6f\n", message.c_str(), v.x, v.y, v.z);
}

// distance > preferredDistance || connected
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

float calculateFriction() {
	long timeFromStart = glutGet(GLUT_ELAPSED_TIME);
	return min(timeFromStart * timeFromStart / 100000000, 1);
}

class Node {
	vec3 position;
	float mass;
	vec3 speed;
	vec3 acceleration;
	unsigned int nodeVBO; //vbo for the nodes

public:

	Node() {
		randomizePosition();
		mass = 1;
		speed = vec3(0, 0, 0);
		acceleration = vec3(0, 0, 0);
	}

	Node(vec3 pos) : Node() {
		position = pos;
	}

	//Hyperbolic coordinates to normalized ones
	// x/z, y/z
	vec2 scaleToNormalized(vec3 hyperbolic) {
		return vec2(hyperbolic.x / hyperbolic.z, hyperbolic.y / hyperbolic.z);
	}

	//Normalized coordinates to hyperbolic ones
	// 1 / (1-z^2)
	vec3 normalizedToHyperbolic(vec3 normalized) {
		return vec3(normalized.x, normalized.y, 1.0f) / sqrt(1.0f - (normalized.x) * (normalized.x) - (normalized.y) * (normalized.y));
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
		this->position = scaleToNormalized(vec3(x, y, z));
	}

	vec3 getPosition() {
		return position;
	}

	void setMass(float mass) {
		this->mass = mass;
	}

	void changePosition(vec3 vector) {
		this->position = this->position + vector;
	}

	void scaleNode(float lambda) {
		this->position = this->position / lambda;
		this->speed = this->speed / lambda;
		this->acceleration = this->acceleration / lambda;
	}

	void move(float deltaTime) {
		this->position = this->position + speed * deltaTime;
	}

	void applyForce(vec3 force, float deltaTime) { // kb 0.000040 nagyságrenű számok
		vec3 deltaSpeed = (force / mass) * deltaTime;
		this->speed = this->speed + deltaSpeed;
		this->speed = this->speed * (1 - calculateFriction());
	}

	float lorentzForce(vec3 p1, vec3 p2) {
		return ((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) - (p1.z - p2.z) * (p1.z - p2.z));
	}

	float getNormalizedDistanceSqr(Node& other) {
		vec3 pos1 = this->getPosition();
		vec3 pos2 = other.getPosition();
		vec3 diff = pos1 - pos2;
		return dot(diff, diff);
	}

	float getNormalizedDistance(Node& other) {
		return sqrtf(getNormalizedDistanceSqr(other));
	}

	float getHyperbolicDistance(Node& other) {
		return acosf(-lorentzForce(this->getPosition(), other.getPosition()));
	}

	float getForceMagnitudeConnected(Node& other) {
		float distance = getNormalizedDistance(other);
		if (distance > preferredDistance) {
			float magnitude = magicFormula(distance - preferredDistance);
			//printf("distance: %9.6f, magnitude: %9.6f\n", distance, magnitude);
			return min(magnitude, forceLimitConnectedPositive);
		}
		else {
			float magnitude = magicFormula2(distance);
			return max(magnitude, -forceLimitConnectedNegative);
		}
	}

	float getForceMagnitudeDisconnected(Node& other) {
		float distance = getNormalizedDistance(other);
		float magnitude = magicFormula3(distance);
		return max(magnitude, -forceLimitDisconnectedNegative);
	}

	void draw() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &nodeVBO);
		glBindBuffer(GL_ARRAY_BUFFER, nodeVBO);

		std::vector<vec2> vertices;

		vertices.push_back(vec2(getPosition().x, getPosition().y));

		/*int sides = 12;
		float radius = 0.05f;

		for (int i = 0; i < sides; i++) {
			float hyperX = ((cosf(360 / sides * i * M_PI / 180) * radius) + getPosition().x - 1);
			float hyperY = ((sinf(360 / sides * i * M_PI / 180) * radius) + getPosition().y - 1);
			vertices.push_back(vec2(hyperX / getPosition().z, hyperY / getPosition().z));
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

		glPointSize(10.0f);
		glBindVertexArray(vao);
		glDrawArrays(GL_POINTS, 0, 1);
		//glDrawArrays(GL_TRIANGLE_FAN, 0, sides);
	}
};

class Edge {
	int node1;
	int node2;
	bool isVisible = false;
	vec3 color = vec3(0, 1, 0);
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

	void draw(std::vector<Node>& nodeList) {
		if (isVisible) {
			glGenVertexArrays(1, &vao);
			glBindVertexArray(vao);

			glGenBuffers(1, &edgeVBO);
			glBindBuffer(GL_ARRAY_BUFFER, edgeVBO);

			std::vector<vec2> vertices;

			vertices.push_back(vec2(nodeList[node1].getPosition().x, nodeList[node1].getPosition().y));
			vertices.push_back(vec2(nodeList[node2].getPosition().x, nodeList[node2].getPosition().y));

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

	int requiredEdgeCount() {
		return (numberOfNodes * (numberOfNodes - 1) / 2) * visibleEdgesInPercent / 100;
	}

	std::vector<Node> getNodes() {
		return nodes;
	}

	std::vector<Edge> getEdges() {
		return edges;
	}

	int calculateDegree(int nodeIndex) {
		int degree = 0;
		for (int i = 0; i < edges.size(); i++) {
			if (edges[i].getNode1() == nodeIndex || edges[i].getNode2() == nodeIndex && edges[i].getVisible()) {
				degree++;
			}
		}
		return degree;
	}

	vec3 calculateHub() {
		vec3 hub = vec3(0, 0, 0);
		for each (Node node in nodes) {
			hub = hub + node.getPosition();
		}
		return hub / nodes.size();
	}

	// finds the farthest node
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

	void resetNodePosition() {
		for each (Node node in nodes) {
			node.randomizePosition();
		}
	}

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
		//printf("visibleEdges: %d", visibleEdgesCount); // to check the number of visible/real edges
	}

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

	void moveTowardsCenter() {
		vec3 hub = -calculateHub();
		for (int i = 0; i < nodes.size(); i++) {
			nodes[i].changePosition(hub);
		}
	}

	void draw() {
		for each (Edge edge in edges)
		{
			edge.draw(nodes);
		}
		for each (Node node in nodes)
		{
			node.draw();
		}
	}

	boolean isConnected(int n1, int n2) {
		for (int i = 0; i < edges.size(); i++)
			if (edges[i].getNode1() == n1 && edges[i].getNode2() == n2 && edges[i].getVisible())
				return true;
		return false;
	}

	vec3 calculateForceFrom(int current, int other) {
		vec3 diff = nodes[other].getPosition() - nodes[current].getPosition();
		if (this->isConnected(current, other))
			return normalize(diff) * nodes[current].getForceMagnitudeConnected(nodes[other]);
		return normalize(diff) * nodes[current].getForceMagnitudeDisconnected(nodes[other]);
	}

	void keepItOnTheScreen() {
		float lambda = findExtremeCoordinates();
		for (int i = 0; i < nodes.size(); i++)
			nodes[i].scaleNode(lambda);
	}

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

	void modifyGraph(float deltaTime) {
		for (int i = 0; i < nodes.size(); i++)
		{
			vec3 force = summariseForces(i);
			nodes[i].applyForce(force, deltaTime);
		}
		for (int i = 0; i < nodes.size(); i++)
			nodes[i].move(deltaTime);
		moveTowardsCenter();
		keepItOnTheScreen();
	}
};

Graph graph;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	/*Node n1 = Node(vec3(-0.5f, 0.0f, 0));
	Node n2 = Node(vec3(0.5f, 0.0f, 0));
	Node n3 = Node(vec3(0.5f, 0.5f, 0));
	Edge e1 = Edge(0, 1, true);
	Edge e2 = Edge(0, 2, true);
	Edge e3 = Edge(2, 1, true);
	graph.addNode(n1);
	graph.addNode(n2);
	graph.addNode(n3);
	graph.addEdge(e1);*/


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
		if (spaceKeyDown) {
			spaceKeyDown = false;
			//graph.resetNodePosition();
		}
		else {
			spaceKeyTime = glutGet(GLUT_ELAPSED_TIME);
			spaceKeyDown = true;
		}
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

	graph.modifyGraph(min(deltaTime, 0.3));
	//printf("%6.4f\n", deltaTime);

	timeAtLastFrame = glutGet(GLUT_ELAPSED_TIME);

	glutPostRedisplay();
}
