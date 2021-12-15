using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class CharacterControl : MonoBehaviour
{

    private Waypoint targetWaypoint;

    public Transform Head;

    private Animator avatar;

    private const float SPEED_DAMP_TIME = .25f;
    private const float DIRECTION_DAMP_TIME = .25f;

    private Vector3 turnDestination;

    /// <summary>
    /// 这个waypoint的作用仅仅是现实角色到第一个waypoint的连线
    /// </summary>
    private Waypoint myFirstWaypoint;

    //private Transform headLookAtWhenTalking;

    //private float lookAtWeight = 0;
    private int moveSpeedHash;
    private float moveSpeed;

    void Awake()
    {
        avatar = GetComponent<Animator>();
        moveSpeedHash = Animator.StringToHash("MoveSpeed");
        UpdateMoveType();
    }

    public void SetTargetWaypoint(Waypoint waypoint)
    {
        targetWaypoint = waypoint;
    }

    private void UpdateMoveType()
    {

        if (targetWaypoint)
        {
            switch (targetWaypoint.Move)
            {
                case Waypoint.MoveType.Walk:
                    moveSpeed = 0f;
                    break;
                case Waypoint.MoveType.Run:
                    moveSpeed = 1f;
                    break;
                default:
                    Debug.LogAssertion("UNDONE");
                    break;
            }
        }
    }

    // 暂时不启用头部的IK功能
    //void OnAnimatorIK(int layerIndex)
    //{
    //    if (headLookAtWhenTalking && avatar.GetCurrentAnimatorStateInfo(0).IsTag("Talking"))
    //    {
    //        avatar.SetLookAtPosition(headLookAtWhenTalking.position);
    //        avatar.SetLookAtWeight(1);
    //    }
    //    else
    //    {
    //        avatar.SetLookAtWeight(0);
    //    }
    //}

    void Update()
    {
        if (avatar == null)
        {
            return;
        }

        avatar.SetFloat(moveSpeedHash, moveSpeed, 0.3f, Time.deltaTime);

        if (targetWaypoint)
        {
            avatar.speed = targetWaypoint.AnimatorSpeed;
        }

        if (avatar.GetCurrentAnimatorStateInfo(0).IsTag("Move")
            && targetWaypoint) // 走到最后一个Waypoint后TargetWaypoint为null，但是walk里的逻辑还尝试访问TargetWaypoint，导致报错
        {
            // 忽略y计算距离
            float distanceX = targetWaypoint.transform.position.x - avatar.rootPosition.x;
            float distanceZ = targetWaypoint.transform.position.z - avatar.rootPosition.z;
            Vector3 waypointPosition = targetWaypoint.transform.position;
            Vector3 characterPosition = avatar.rootPosition;
            //float distanceToTarget = Vector3.SqrMagnitude(targetWaypoint.transform.position - avatar.rootPosition);
            float distanceToTarget = Mathf.Sqrt(distanceX * distanceX + distanceZ * distanceZ);
            if (distanceToTarget > targetWaypoint.Range)
            {
                Vector3 curentDir = avatar.rootRotation * Vector3.forward;
                Vector3 wantedDir = (targetWaypoint.transform.position - avatar.rootPosition).normalized;

                if (Vector3.Dot(curentDir, wantedDir) > 0)
                {
                    // -90 ~ 90
                    avatar.SetFloat("MoveDirection", Vector3.Cross(curentDir, wantedDir).y, DIRECTION_DAMP_TIME, Time.deltaTime);
                }
                else
                {

                    avatar.SetFloat("MoveDirection", Vector3.Cross(curentDir, wantedDir).y > 0 ? 1 : -1, DIRECTION_DAMP_TIME, Time.deltaTime);
                }
            }
            else
            {
                if (targetWaypoint.AnimationToPlay != Waypoint.AnimationEnum.None)
                {
                    //avatar.SetInteger("ClipID", TargetWaypoint.ClipID);
                    avatar.SetBool("Turn", true);
                }

                if (targetWaypoint.BodyLookAt)
                {
                    // TODO: turnDestination为0向量的保护
                    turnDestination = targetWaypoint.BodyLookAt.position - targetWaypoint.transform.position;
                    turnDestination.y = 0;
                }
                else
                {
                    // 如果没有LookAtTarget，则旋转目标就是Avatar的当前朝向
                    turnDestination = avatar.rootRotation * Vector3.forward;
                    turnDestination.y = 0;
                }
                //headLookAtWhenTalking = TargetWaypoint.HeadLookAt;
                if (targetWaypoint.Next)
                {
                    Waypoint.AnimationEnum previousAnimation = targetWaypoint.AnimationToPlay;
                    targetWaypoint = targetWaypoint.Next;
                    if (previousAnimation == Waypoint.AnimationEnum.None)
                    {
                        // 如果播动画，则调用UpdateMoveType()的时机在avatar.SetBool("Turn", false);时
                        // 如果不播动画，则在这里调用UpdateMoveType()
                        UpdateMoveType();
                    }
                }
                else
                {
                    targetWaypoint = null;
                    avatar.SetBool("End", true);
                }
            }
        }
        else if (avatar.GetCurrentAnimatorStateInfo(0).IsTag("Turn"))
        {
            avatar.speed = 1f;
            Vector3 curentDir = avatar.rootRotation * Vector3.forward;

            float angle = Vector3.Angle(curentDir, turnDestination);

            if (angle < 5)
            {
                avatar.SetBool("Turn", false);
                if (targetWaypoint)
                {
                    Vector3 wantedDir = (targetWaypoint.transform.position - avatar.rootPosition).normalized;
                    avatar.SetFloat("MoveStartAngle", (Vector3.Cross(curentDir, wantedDir).y > 0 ? 1 : -1) * Vector3.Angle(curentDir, wantedDir), 0, Time.deltaTime);
                }
                // 如果在avatar.SetBool("Turn", true)时就掉用UpdateMoveType()，则跑走的最后几帧速度就不对了
                UpdateMoveType();
            }
            else
            {
                avatar.SetFloat("TurnDirection", (Vector3.Cross(curentDir, turnDestination).y > 0 ? 1 : -1) * angle, 0, Time.deltaTime);
            }
        }
        else if (avatar.GetCurrentAnimatorStateInfo(0).IsTag("Talking"))
        {

            avatar.speed = 1f;
        }
        else
        {
            avatar.speed = 1f;
        }
    }

    void LateUpdate()
    {
        // HACK:不知道为什么，有时候角色得高度会改变，这里强制让角色始终保持高度一致
        Vector3 position = this.transform.position;
        position.y = 0;
        this.transform.position = position;
    }
}
