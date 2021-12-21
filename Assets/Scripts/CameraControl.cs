using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraControl : MonoBehaviour
{
    public Transform FocusTarget;
    private Vector3 target2Me;

    // Start is called before the first frame update
    void Awake()
    {
        target2Me = this.transform.position - FocusTarget.position;
    }

    // Update is called once per frame
    void LateUpdate()
    {
        this.transform.position = FocusTarget.position + target2Me;
        this.transform.LookAt(FocusTarget.position);
    }
}