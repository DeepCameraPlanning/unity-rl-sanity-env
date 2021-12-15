/*
Requirements:

    - Install `ml-agent` through the Package Manager

    - Add this script to the agent (camera)
        + Set the target.s

    - Add a `Decision Requester` to the agent
        + Set the decision period

    - Add a `Behavior Parameter`
        + Set up the name: `CameraControl`
        + Set the observation space:
            * Space size = x
            * Stacked vector = 1
        + Set the action space:
            * Continuous actions = 0
            * Discrete actions = 2 (L, R)
*/

using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using UnityEngine;
using UnityEngine.Playables;
using UnityEngine.SceneManagement;
using UnityEditor;
using System.IO;
using System.Threading;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class CameraAgent : Agent
{

    private Camera main;
    public Transform target;
    public Transform obstacle;

    private RenderTexture renderTexture;
    private Texture2D screenShot;

    private float obstacleTheta, mainTheta;
    private float obstacleDirection, mainDirection;

    int timeStep = 30;
    private float rewardCollision;

    EnvironmentParameters resetParams;

    public override void Initialize()
    {
        // Initialize env
        resetParams = Academy.Instance.EnvironmentParameters;
        main = gameObject.GetComponent<Camera>();

        renderTexture = new RenderTexture(Screen.width, Screen.height, 24);
        screenShot = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);

    }

    public override void OnEpisodeBegin()
    {
        // Initialize the camera position
        this.initializeScene();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Get camera and obstacle positions
        sensor.AddObservation(main.transform.localPosition);
        sensor.AddObservation(obstacle.localPosition);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        obstacleDirection = 0.006f;

        int action = actionBuffers.DiscreteActions[0];
        if (action == 0) mainDirection = -0.03f;
        else mainDirection = 0.03f;
        updateState();

        if (rewardCollision == -1)
        {
            SetReward(-1.0f);
            EndEpisode();
        }
        else SetReward(1.0f);

        timeStep += 1;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // // Heuristic method to test the env
        // var discreteActionsOut = actionsOut.DiscreteActions;
        // if (Input.GetKey("left"))
        // {
        //     discreteActionsOut[0] = 1;

        // }
        // if (Input.GetKey("right"))
        // {
        //     discreteActionsOut[0] = 0;
        // }
        // discreteActionsOut[0] = 1;
    }


    private void initializeScene()
    {
        rewardCollision = -1;
        while (rewardCollision == -1)
        {
            obstacleTheta = Random.Range(0, 2f);
            mainTheta = Random.Range(0, 2f);

            updateState();
        }
    }

    private void updateState()
    {
        Vector3 head = target.transform.position;

        obstacleTheta += obstacleDirection;
        mainTheta += mainDirection;

        obstacle.transform.position = head + 2f * new Vector3(Mathf.Cos(obstacleTheta), 0, Mathf.Sin(obstacleTheta));
        transform.position = head + 5f * new Vector3(Mathf.Cos(mainTheta), 0, Mathf.Sin(mainTheta));

        transform.LookAt(target.transform.position);

        if (isColliding()) rewardCollision = -1;
        else rewardCollision = 1;
    }

    private bool isColliding()
    {
        Vector3 head = target.transform.position;
        Vector3 hc = obstacle.transform.position - head;
        Vector3 hd = transform.position - head;

        float angle = Vector3.Angle(hc, hd);

        Debug.DrawRay(head, hc);
        Debug.DrawRay(head, hd);

        return angle < 45f;
    }
}