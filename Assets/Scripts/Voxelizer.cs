using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;
using UnityEditor;

public class Voxelizer : MonoBehaviour
{

    public bool showPointGrid = false;

    private BoxCollider areaToVoxelizeCollider;

    public ComputeShader voxelizeAreaCS;

    public ComputeShader sdfCS;
    
    ComputeBuffer vertexBuffer;
    ComputeBuffer vertexCountBuffer;

    ComputeBuffer trianglesBuffer;
    ComputeBuffer trianglesCountBuffer;

    ComputeBuffer objLocalToGridMatrix;

    public float cellHalfSize;

    List<GameObject> previousGOInVoxelizedArea;

    bool noObjInCollider = false;

    public bool computeSDF = true;

    // private MeshRenderer renderer;

    private Material material;

    public GameObject planePrefab;

    public Texture3D textureGrid;

    public Shader volumeVisu;

    private Vector3 voxelCount;

    public Gradient customGradiantMap;

    public float displayAlpha = 0.2f;

    public int SDFKernelSize = 3;

    public LayerMask m_LayerMask;

    public GameObject target;

    public float[] outArray;
    public Vector3 outDim;

    

// float4 _BoundsMin;
// float _CellHalfSize;
// int _GridWidth;
// int _GridHeight;
// int _GridDepth;

    // Start is called before the first frame update
    void Start()
    {

        createGradiantMap();

        areaToVoxelizeCollider = gameObject.GetComponent<BoxCollider>();

        voxelCount = areaToVoxelizeCollider.size / 2f / cellHalfSize;
        // Debug.Log(voxelCount.ToString("f16"));
        int xGridSize = Mathf.CeilToInt(voxelCount.x);
        int yGridSize = Mathf.CeilToInt(voxelCount.y);
        int zGridSize = Mathf.CeilToInt(voxelCount.z);

        TextureFormat format = TextureFormat.RGBA32;
        TextureWrapMode wrapMode =  TextureWrapMode.Clamp;

        // Create the texture and apply the configuration
        textureGrid = new Texture3D(xGridSize, yGridSize, zGridSize, format, false);
        textureGrid.wrapMode = wrapMode;
        Debug.Log(SystemInfo.supportsComputeShaders);

        //Material m = gameObject.GetComponent<MeshRenderer>().material;
        //m.shader = volumeVisu;
        //m.SetTexture("_MainTex", textureGrid);
        
        
        //setVisualisationPlane();
    }

    // Update is called once per frame
    void Update()
    {
        gameObject.transform.position = target.transform.position;
        gameObject.transform.rotation = target.transform.rotation;
        if (needRevoxelization()) {

            List<GameObject> objToVoxelizeList = getGameObjectInVoxelizedArea();
            
            //Mesh a = objToVoxelizeList[0].GetComponent<MeshFilter>().sharedMesh;
            
            //setUpVertexAndTrianglCB(objToVoxelizeList);
            gridSetup();
            List<Mesh> meshes = getMeshes(objToVoxelizeList);

            fillVoxelBufferWithGPU(objToVoxelizeList, meshes);

            updateTexture();
        }
    }


    private void createGradiantMap () {
        customGradiantMap = new Gradient();
        
        int colorKeyNumber = 8;

        GradientColorKey[] colorKey = new GradientColorKey[colorKeyNumber];
        GradientAlphaKey[] alphaKey = new GradientAlphaKey[2];
        
        alphaKey[0].alpha = 1.0f;
        alphaKey[0].time  = 0.0f;
        alphaKey[1].alpha = 1.0f;
        alphaKey[1].time  = 1.0f;

        for (int i = 0; i < colorKeyNumber; i++) {
            float time = (float)i/(float)colorKeyNumber;
            colorKey[i].color = ColorPalette.TurboColorMap(time);
            colorKey[i].time = time;
        }
        //Debug.Log("_____________________________________________");
        customGradiantMap.SetKeys(colorKey, alphaKey);
    }

    private void setUpVertexAndTrianglCB(List<GameObject> objToVoxelize) {
        if (objBufferHasBeenModified()) return; 
        //float t = Time.realtimeSinceStartup;

        // counting elems
        int [] vertexNumber = new int[objToVoxelize.Count];
        int [] trianglesNumber = new int[objToVoxelize.Count];
        int totalVertexNumber = 0;
        int totalTrianglesNumber = 0;
        
        List<Vector3> vertexArr = new List<Vector3> ();
        List<int> trianglesArr = new List<int> ();
        List<Matrix4x4> objToGridMatrix = new List<Matrix4x4>();

        for (int i = 0; i < objToVoxelize.Count; i++) {
            MeshFilter mf = objToVoxelize[i]?.GetComponent<MeshFilter>();
            Mesh m;
            if (mf == null) m = objToVoxelize[i].GetComponent<SkinnedMeshRenderer>()?.sharedMesh;
            else m = mf.sharedMesh;

            // if (mf == null) m = objToVoxelize[i].GetComponent<MeshCollider>().sharedMesh; 
            // else m = mf.sharedMesh;

            int vertN = m.vertexCount;
            int triN = m.triangles.Length / 3;

            totalVertexNumber += vertN;
            totalTrianglesNumber += triN;

            vertexNumber[i] = vertN;
            trianglesNumber[i] = triN;

            //var tmp = new List<Vector3>
            vertexArr.AddRange(m.vertices);
            trianglesArr.AddRange(m.triangles);

            objToGridMatrix.Add(objToVoxelize[i].transform.localToWorldMatrix * gameObject.transform.worldToLocalMatrix);
        }

        if (objToVoxelize.Count > 0) {
            vertexCountBuffer = new ComputeBuffer(vertexNumber.Length, sizeof(int));
            vertexCountBuffer.SetData(vertexNumber);

            vertexBuffer = new ComputeBuffer (vertexArr.Count, sizeof(float) * 3);
            vertexBuffer.SetData(vertexArr.ToArray());

            trianglesCountBuffer = new ComputeBuffer(trianglesNumber.Length, sizeof(int));
            trianglesCountBuffer.SetData(trianglesNumber);

            trianglesBuffer = new ComputeBuffer(trianglesArr.Count, sizeof(int));
            trianglesBuffer.SetData(trianglesArr.ToArray());
            
            objLocalToGridMatrix = new ComputeBuffer(objToVoxelize.Count, sizeof(float) * 4 * 4, ComputeBufferType.Default);
            objLocalToGridMatrix.SetData(objToGridMatrix.ToArray());

            noObjInCollider = false;

        } else {
            noObjInCollider = true;
        }

        gridSetup();
    
        //Debug.Log("time test : " + (Time.realtimeSinceStartup - t).ToString("f16"));
    }

    
    private void setVisualisationPlane() {

        Vector3 center = areaToVoxelizeCollider.center;
        Vector3 s = areaToVoxelizeCollider.size / 2f;
        Vector3 boundsMin = center - s;
        Vector3 scale = areaToVoxelizeCollider.size / 10f;
        
        int xGridSize = Mathf.CeilToInt(areaToVoxelizeCollider.size.x);
        int yGridSize = Mathf.CeilToInt(areaToVoxelizeCollider.size.y);
        int zGridSize = Mathf.CeilToInt(areaToVoxelizeCollider.size.z);

        float cellSize = cellHalfSize*2f;

        Quaternion q0 = Quaternion.Euler(0,0,0);
        Quaternion q1 = Quaternion.Euler(180,0,0);

        for (int i = 0; i < xGridSize; i++) {
            Vector3 pos = center;
            pos.y = boundsMin.y + i;
            GameObject p1 = GameObject.Instantiate(planePrefab, pos, q0);
            p1.transform.localScale = new Vector3(scale.x, 1, scale.z);
            //p1.transform.localScale = new Vector3(4, 1, 4);
            p1.transform.parent = gameObject.transform;
            Material m1 = p1.GetComponent<MeshRenderer>().material;
            m1.shader = volumeVisu;
            m1.SetTexture("_MainTex", textureGrid);

            GameObject p2 = GameObject.Instantiate(planePrefab, pos, q1);
            p2.transform.localScale = new Vector3(scale.x, 1, scale.z);
            p2.transform.parent = gameObject.transform;
            Material m2 = p2.GetComponent<MeshRenderer>().material;
            m2.shader = volumeVisu;
            m2.SetTexture("_MainTex", textureGrid);
        }
        
        
        // Quaternion q2 = Quaternion.Euler(0,0,90);
        // Quaternion q3 = Quaternion.Euler(0,0,-90);

        // for (int i = 0; i < yGridSize; i++) {
        //     Vector3 pos = center;
        //     pos.x = boundsMin.x + cellSize * i;
        //     GameObject p1 = GameObject.Instantiate(planePrefab, pos, q2);
        //     //p1.transform.localScale = new Vector3(scale.y, 1, scale.z);
        //     p1.transform.parent = gameObject.transform;
        //     Material m1 = p1.GetComponent<MeshRenderer>().material;
        //     m1.shader = volumeVisu;
        //     m1.SetTexture("_MainTex", textureGrid);

        //     GameObject p2 = GameObject.Instantiate(planePrefab, pos, q3);
        //     //p2.transform.localScale = new Vector3(scale.y, 1, scale.z);
        //     p2.transform.parent = gameObject.transform;
        //     Material m2 = p2.GetComponent<MeshRenderer>().material;
        //     m2.shader = volumeVisu;
        //     m2.SetTexture("_MainTex", textureGrid);
        // }

        // Quaternion q4 = Quaternion.Euler(90,0,0);
        // Quaternion q5 = Quaternion.Euler(-90,0,0);

        // for (int i = 0; i < zGridSize; i++) {
        //     Vector3 pos = center;
        //     pos.z = boundsMin.z + cellSize * i;
        //     GameObject p1 = GameObject.Instantiate(planePrefab, pos, q4);
        //     //p1.transform.localScale = new Vector3(scale.x, 1, scale.y);
        //     p1.transform.parent = gameObject.transform;
        //     Material m1 = p1.GetComponent<MeshRenderer>().material;
        //     m1.shader = volumeVisu;
        //     m1.SetTexture("_MainTex", textureGrid);

        //     GameObject p2 = GameObject.Instantiate(planePrefab, pos, q5);
        //     //p2.transform.localScale = new Vector3(scale.x, 1, scale.y);
        //     p2.transform.parent = gameObject.transform;
        //     Material m2 = p2.GetComponent<MeshRenderer>().material;
        //     m2.shader = volumeVisu;
        //     m2.SetTexture("_MainTex", textureGrid);
        // }

    }

    private void gridSetup() {
        voxelCount = areaToVoxelizeCollider.size / 2f / cellHalfSize;
    }

    private List<Mesh> getMeshes(List<GameObject> objToVoxelize) {
        List<Mesh> res = new List<Mesh>();
        if (objBufferHasBeenModified()) return res;
        
        for (int i = 0; i < objToVoxelize.Count; i++) {
            MeshFilter mf = objToVoxelize[i]?.GetComponent<MeshFilter>();
            Mesh m;
            if (mf == null) m = objToVoxelize[i].GetComponent<SkinnedMeshRenderer>()?.sharedMesh;
            else m = mf.sharedMesh;

            res.Add(m);
        }

        return res;
    }

    bool objBufferHasBeenModified () {
        // TODO
        return false;
    }

    bool needRevoxelization () {
        // TODO
        return true;
    }

    List<GameObject> getGameObjectInVoxelizedArea () {
        
        gameObject.SetActive(false);

        Bounds b = areaToVoxelizeCollider.bounds;

        // Debug.Log(b.center.ToString("f5"));
        
        // Debug.Log(areaToVoxelizeCollider.size.ToString("f5"));

        Collider[] ObjToVoxelize = Physics.OverlapBox(b.center, areaToVoxelizeCollider.size/2f, transform.rotation, m_LayerMask);

        gameObject.SetActive(true);

        List<GameObject> res = new List<GameObject>();

        //Debug.Log("Object founded in the volizer area = " + ObjToVoxelize.Length);
        foreach(Collider c in ObjToVoxelize) {
            res.Add(c.gameObject);
        }

        return res;
    }

    int test = 0;
    void fillVoxelBufferWithGPU (List<GameObject> objToVoxelizeList, List<Mesh> meshes) {
        
        gridSetup();
        Matrix4x4 gridW2L = gameObject.transform.worldToLocalMatrix;
        int xGridSize = Mathf.CeilToInt(voxelCount.x);
        int yGridSize = Mathf.CeilToInt(voxelCount.y);
        int zGridSize = Mathf.CeilToInt(voxelCount.z);

        int gSize = xGridSize*yGridSize*zGridSize;
        
        ComputeBuffer voxelsBuffer = new ComputeBuffer(gSize, sizeof(float));
        float[] vArr = new float[gSize]; 
        voxelsBuffer.SetData(vArr);
        
        ComputeBuffer sdfBuffer = new ComputeBuffer(gSize, sizeof(float));
        float[] sdfArr = new float[gSize]; 
        sdfBuffer.SetData(sdfArr);

        for (int i = 0; i < objToVoxelizeList.Count; i++) {

            // Debug.Log(objToVoxelizeList[i].name + " : " + test++);
            var voxelizeKernel = voxelizeAreaCS.FindKernel("VoxelizeMesh");
            voxelizeAreaCS.SetInt("_GridWidth", xGridSize);
            voxelizeAreaCS.SetInt("_GridHeight", yGridSize);
            voxelizeAreaCS.SetInt("_GridDepth", zGridSize);

            voxelizeAreaCS.SetFloat("_CellHalfSize", cellHalfSize);
            voxelizeAreaCS.SetBool("noObjInCollider", false);

            voxelizeAreaCS.SetMatrix("_WorldToGridMatrix", gridW2L);
            voxelizeAreaCS.SetBuffer(voxelizeKernel, "occupancy_grid", voxelsBuffer);

            Mesh m = meshes[i];
            int vertnum = m.vertices.Length;
            ComputeBuffer vertCB = new ComputeBuffer(vertnum, sizeof(float)*3);
            vertCB.SetData(m.vertices);

            int trinum = m.triangles.Length;
            ComputeBuffer triCB = new ComputeBuffer(trinum, sizeof(int));
            triCB.SetData(m.triangles);

            // Debug.Log("Vert num = " + vertnum);
            // Debug.Log("Tri num = " + trinum/3);

            voxelizeAreaCS.SetBuffer(voxelizeKernel, "vertices_array", vertCB);
            voxelizeAreaCS.SetBuffer(voxelizeKernel, "triangles_array", triCB);
            voxelizeAreaCS.SetInt("triangles_number", trinum);

            Matrix4x4 objL2W = objToVoxelizeList[i].transform.localToWorldMatrix;

            voxelizeAreaCS.SetMatrix("_ObjectToWorldMatrix",  objL2W);

            Vector3 center = areaToVoxelizeCollider.center;
            Vector3 s = areaToVoxelizeCollider.size / 2f;
            voxelizeAreaCS.SetVector("_BoundsMin", center - s);

            voxelizeAreaCS.GetKernelThreadGroupSizes(voxelizeKernel, out uint xGroupSize, out uint yGroupSize, out uint zGroupSize);

            voxelizeAreaCS.Dispatch(voxelizeKernel,
                Mathf.CeilToInt(xGridSize / (float) xGroupSize),
                Mathf.CeilToInt(yGridSize / (float) yGroupSize),
                Mathf.CeilToInt(zGridSize / (float) zGroupSize));
            

            //_gridPointCount = voxelsBuffer.count;

            var volumeKernel = voxelizeAreaCS.FindKernel("FillVolume");

            

            voxelizeAreaCS.SetBuffer(volumeKernel, "occupancy_grid", voxelsBuffer);
            voxelizeAreaCS.SetBuffer(volumeKernel, "vertices_array", vertCB);
            voxelizeAreaCS.SetBuffer(volumeKernel, "triangles_array", triCB);
            voxelizeAreaCS.SetInt("triangles_number", trinum);
            voxelizeAreaCS.SetMatrix("_ObjectToWorldMatrix", objL2W);
            voxelizeAreaCS.GetKernelThreadGroupSizes(volumeKernel, out xGroupSize, out yGroupSize, out zGroupSize);

            voxelizeAreaCS.Dispatch(volumeKernel,
                Mathf.CeilToInt(xGridSize / (float) xGroupSize),
                Mathf.CeilToInt(yGridSize / (float) yGroupSize),
                Mathf.CeilToInt(zGridSize / (float) zGroupSize));
            
            if (computeSDF){
                var SDFKernel = sdfCS.FindKernel("Conv3D");

                sdfCS.SetInt("_GridWidth", xGridSize);
                sdfCS.SetInt("_GridHeight", yGridSize);
                sdfCS.SetInt("_GridDepth", zGridSize);

                sdfCS.SetInt("kernelSize", SDFKernelSize);

                sdfCS.SetBuffer(SDFKernel, "occupancy_grid", voxelsBuffer); 

                sdfCS.SetBuffer(SDFKernel, "SDF", sdfBuffer); 

                sdfCS.GetKernelThreadGroupSizes(SDFKernel, out xGroupSize, out yGroupSize, out zGroupSize);
                sdfCS.Dispatch(SDFKernel,
                    Mathf.CeilToInt(xGridSize / (float) xGroupSize),
                    Mathf.CeilToInt(yGridSize / (float) yGroupSize),
                    Mathf.CeilToInt(zGridSize / (float) zGroupSize));
            }

            
            vertCB.Dispose();
            triCB.Dispose();

        }

        outArray = new float[xGridSize*yGridSize*zGridSize];
        outDim   = new Vector3(xGridSize, yGridSize, zGridSize);

        if (computeSDF) {
            sdfBuffer.GetData(outArray);
        } else {
            voxelsBuffer.GetData(outArray);
        }

        if (showPointGrid) 
            simpleDisplay(xGridSize, yGridSize, zGridSize);
        
        sdfBuffer.Dispose();
        voxelsBuffer.Dispose();
    }

    private void updateTexture()
    {
        int xGridSize = Mathf.CeilToInt(voxelCount.x);
        int yGridSize = Mathf.CeilToInt(voxelCount.y);
        int zGridSize = Mathf.CeilToInt(voxelCount.z);

        // Create a 3-dimensional array to store color data
        int arrSize = xGridSize * yGridSize * zGridSize;
        Color[] colors = new Color[arrSize];

        float[] array = outArray;

        Vector3 center = areaToVoxelizeCollider.center;
        Vector3 s = areaToVoxelizeCollider.size / 2f;
        Vector3 boundsMin = center - s;
        float inverseResolutionx = 1.0f / (xGridSize - 1.0f);
        float inverseResolutiony = 1.0f / (yGridSize - 1.0f);
        float inverseResolutionz = 1.0f / (zGridSize - 1.0f);
        for (int i = 0; i < xGridSize; i++) {
            for (int j = 0; j < yGridSize; j++) {
                for (int k = 0; k < zGridSize; k++) {
                    int voxelIndex = i + xGridSize * (j + yGridSize * k);
                    colors[voxelIndex] = ColorPalette.TurboColorMap(array[voxelIndex]);
                    colors[voxelIndex].a = array[voxelIndex];
                    // colors[voxelIndex] = new Color(
                    //     i * inverseResolutionx,
                    //     j * inverseResolutiony, 
                    //     k * inverseResolutionz, 
                    //     1.0f
                    // );
                }
            }
        }

        // Copy the color values to the texture
        textureGrid.SetPixels(colors);

        // Apply the changes to the texture and upload the updated texture to the GPU
        textureGrid.Apply();        

        // Save the texture to your Unity Project
        // AssetDatabase.CreateAsset(texture, "Assets/Example3DTexture.asset");
    }

    void simpleDisplay (int xGroupSize, int yGroupSize, int zGroupSize) {

        float[] grid = outArray;
        

        //Debug.Log("grid[0] = " + grid[0]);

        Vector3 center = areaToVoxelizeCollider.center;
        Vector3 s = areaToVoxelizeCollider.size / 2f;
        Vector3 boundsMin = center - s;
        float sumX = 0;
        float sumY = 0;
        float sumZ = 0;
        int ct=0;
        for (int i = 0; i < xGroupSize; i++) {
            for (int j = 0; j < yGroupSize; j++) {
                for (int k = 0; k < zGroupSize; k++) {
                    int voxelIndex = i + xGroupSize * (j + yGroupSize * k);
                    
                    float cellSize = 2 * cellHalfSize;
                    Vector3 position = gameObject.transform.TransformPoint(new Vector3(
                        boundsMin.x + i * cellSize,
                        boundsMin.y + j * cellSize,
                        boundsMin.z + k * cellSize
                    ));

                    // sumX+=i*grid[voxelIndex];
                    // sumY+=j*grid[voxelIndex];
                    // sumZ+=k*grid[voxelIndex];
                    // if(grid[voxelIndex]!=0)
                    //   ct+=1;
                      // Debug.Log(grid[voxelIndex]);

                    
                    
                    //if (occupancyGrid[voxelIndex] > 0) c = Color.red;
                    //Debug.DrawRay(position, Vector3.forward * 0.01f, c);
                    //Debug.Log(occupancyGrid[voxelIndex]);
                    // if (grid[voxelIndex] > 0) {
                        // Debug.Log("grid pos : (" + i+","+j+","+k+",)");
                //         Debug.DrawRay(position, Vector3.forward * 0.01f, ColorPalette.TurboColorMap(grid[voxelIndex]));
                    // }
                }
            }
        }

        // Vector3 CM = new Vector3(sumX/(float)ct, 
        //                          sumY/(float)ct, 
        //                          sumZ/(float)ct);

        // Debug.Log("grid pos : (" + CM[0]+","+CM[1]+","+CM[2]+",)");

    }
    

}

#if UNITY_EDITOR
// Vizer
[CustomEditor(typeof(Voxelizer))]
public class VoxelizerEditor : Editor
{
    // Custom in-scene UI for when ExampleScript
    // component is selected.
    public void OnSceneGUI()
    {
        var t = target as Voxelizer;
        var tr = t.transform;
        var pos = tr.position;

        GameObject voxelArea = t.gameObject;

        BoxCollider collider = voxelArea.GetComponent<BoxCollider>();

        Vector3 scale = t.transform.localScale;

        t.transform.localScale = collider.bounds.size;

        Matrix4x4 matrix = t.transform.localToWorldMatrix;

        t.transform.localScale = scale;

        Handles.matrix = matrix;

        // display an orange disc where the object is
        var color = new Color(1, 0.8f, 0.4f, 1);
        //Handles.color = color;
        //Handles.DrawWireCube(collider.bounds.center, new Vector3(1,1,1));
        // display object "value" in scene
        // GUI.color = color;
        // Handles.Label(pos, t.value.ToString("F1"));

        if (t.textureGrid != null) {
            //Debug.Log("test");
            bool useGradient = t.customGradiantMap != null; 
            Handles.DrawTexture3DVolume(t.textureGrid, 1, t.displayAlpha, FilterMode.Bilinear, useGradient, t.customGradiantMap);
        }
    }
}
#endif  // UNITY_EDITOR


