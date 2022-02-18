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
    public GameObject prefabModel;
    public float modelSize;
    public Material mat;
    GameObject mainModel;
    // public GameObject Cube;
    private float obstacleTheta, mainTheta;
    private Vector3 obstacleRelative, mainRelative;

    // radians
    private float obstacleAmplitude = 2.0f, mainAmplitude = 5.0f;
    // Keep `obstacleDirection = 1`, as the unit direction
    private float obstacleDirection = 1.0f, mainDirection = 2.0f;
    // Divide direction variables by a speed norm in order to normalize them
    private float obstacleSpeed, mainSpeed;
    private const float speedNorm = 500.0f;

    private float rewardCollision;

    public int recordedStep=0;
    EnvironmentParameters resetParams;

    public int cubePolicy = 0;
    
    public float timeScaleValue=1.0f;
    
    public float RaycastDist = 5.0f;

    public override void Initialize()
    {
        // Initialize env
        resetParams = Academy.Instance.EnvironmentParameters;
        main = gameObject.GetComponent<Camera>();

        // Application.targetFrameRate = 30;
        // renderTexture = new RenderTexture(Screen.width, Screen.height, 24);
        // screenShot = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);

        obstacleSpeed = obstacleDirection / speedNorm;
        mainSpeed = mainDirection / speedNorm;

        mainModel = Instantiate(prefabModel, transform.position, transform.rotation);
        mainModel.GetComponentsInChildren<Renderer>().ToList().ForEach(M =>
        {
            M.sharedMaterial = mat;
            M.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            M.receiveShadows = false;
        });
        mainModel.transform.localScale = Vector3.one * modelSize;
        Time.timeScale=timeScaleValue;
        Application.runInBackground=true;
        // Display.displays[0].Deactivate();
    }

    public override void OnEpisodeBegin()
    {
        // Initialize the camera and cube position
        this.initializeScene();
        recordedStep=0;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Get camera and obstacle relative positions
        sensor.AddObservation(mainRelative);
        // sensor.AddObservation(obstacleRelative);
        
        // Debug.Log("cube pos : (" + CM[0]+","+CM[1]+","+CM[2]+",)");

        var vox = GameObject.Find("OccupancyGrid").GetComponent<Voxelizer>();
        float[] outArray = vox.outArray;
        Vector3 outDim = vox.outDim;
        // Debug.Log("received [" + outArray.Length.ToString() + "] elements in the array");
        sensor.AddObservation(outDim);
        // Debug.Log("dim: "+ outDim.ToString());
        // Console.WriteLine("dim: "+ outDim.ToString());
        sensor.AddObservation(outArray);
    }

    // cube Policy from simulator -> server

    // policy 0 -> clockwise (as training)
    // policy 1 -> anti-clockwise (testing balance)

    // policy 2 -> moving speed level 1 (slowest)
    // policy 3 -> moving speed level 2 
    // policy 4 -> moving speed level 3 
    // policy 5 -> moving speed level 4 (fastest) 
    // policy 6 -> random walk??
    public void updateCubeSpeed()
    {
      // Update obstacle angle
      int veloAmp=100;
      switch (cubePolicy)
      {
        case 0:
            obstacleSpeed = obstacleDirection / speedNorm;
          break;

        case 1:
            obstacleSpeed = - obstacleDirection / speedNorm;
          break;

        case 2:
            obstacleSpeed = veloAmp*Mathf.Sin(recordedStep/(speedNorm*20f)*Mathf.PI)/speedNorm;
          break;

        case 3:
            obstacleSpeed = veloAmp*Mathf.Sin(recordedStep/(speedNorm*15f)*Mathf.PI)/speedNorm;
          break;

        case 4:
            obstacleSpeed = veloAmp*Mathf.Sin(recordedStep/(speedNorm*12f)*Mathf.PI)/speedNorm;
          break;

        case 5:
            obstacleSpeed = veloAmp*Mathf.Sin(recordedStep/(speedNorm*10f)*Mathf.PI)/speedNorm;
          break;

        case 6:
            obstacleSpeed = veloAmp*Mathf.Sin(recordedStep/10f*Mathf.PI)/speedNorm;
          break;
      }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        recordedStep+=1;
        updateCubeSpeed();
        // Update obstacle angle
        obstacleTheta += obstacleSpeed;

        // Update camera angle according to the received action
        // Discrete agent
        int action = actionBuffers.DiscreteActions[0];
        if (action == 0) mainTheta -= mainSpeed;
        else if (action == 1) mainTheta += mainSpeed;
        else if (action == 2) mainTheta = mainTheta;
        // Continuous agent
        // mainTheta += mainSpeed * actionBuffers.ContinuousActions[0];

        updateState(mainTheta, obstacleTheta, false);

        if (rewardCollision == -1)
        {
            SetReward(-1.0f);
            EndEpisode();
        }
        else SetReward(1.0f);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // // Heuristic method to test the env
        var discreteActionsOut = actionsOut.DiscreteActions;
        if (Input.GetKey("left"))
        {
            discreteActionsOut[0] = 1;

        }
        else if (Input.GetKey("right"))
        {
            discreteActionsOut[0] = 0;
        }
        else discreteActionsOut[0] = 2;
        // var continuousActionsOut = actionsOut.ContinuousActions;
        // continuousActionsOut[0] = Input.GetAxis("Horizontal");
    }

    private void initializeScene()
    {
        rewardCollision = -1;
        while (rewardCollision == -1)
        {
            obstacleTheta = Random.Range(0, 1.0f);
            mainTheta = Random.Range(0, 1.0f);

            updateState(mainTheta, obstacleTheta, true);
        }
    }

    private void updateState(float mainAngle, float obstacleAngle, bool init)
    {
        Vector3 head = target.transform.position;

        // Compute obstacle and camera phases and relative positions
        float obstaclePhase = 2 * Mathf.PI * obstacleAngle;
        obstacleRelative = new Vector3(Mathf.Cos(obstaclePhase), 0, Mathf.Sin(obstaclePhase));
        float mainPhase = 2 * Mathf.PI * mainAngle;
        mainRelative = new Vector3(Mathf.Cos(mainPhase), 0, Mathf.Sin(mainPhase));

        // Update obstacle and camera positions
        obstacle.transform.position = head + obstacleAmplitude * obstacleRelative;
        transform.position          = head + mainAmplitude * mainRelative;

        transform.LookAt(target.transform.position);
        mainModel.transform.position = transform.position;
        mainModel.transform.rotation = transform.rotation;
        
        if(init)
        {
          if (isColliding()) 
          {
            rewardCollision = -1;
          }
          else 
          {
            rewardCollision = 1;
          }
        }
        else{
          if (isCollidingRC()) 
          {
            rewardCollision = -1;
          }
          else 
          {
            rewardCollision = 1;
          }
        }
        // isCollidingRC();
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

    private bool isCollidingRC()
    {
        RaycastHit hit;
        // Bit shift the index of the layer (3) to get a bit mask
        int layerObs = 1 << 3;

        Vector3 dirRC = (target.transform.position - transform.position).normalized;
        if (Physics.Raycast(transform.position, dirRC, out hit, RaycastDist, layerObs))
        {
            Debug.DrawRay(transform.position, dirRC * hit.distance, Color.red);
            // Debug.Log("Did Hit");
            return true;
        }
        else
        {
            Debug.DrawRay(transform.position, dirRC * RaycastDist, Color.green);
            // Debug.Log("Did not Hit");
            return false;
        }
    }
}